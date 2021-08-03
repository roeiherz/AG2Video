import matplotlib
matplotlib.use('Agg')
import logging
from functools import partial
import os
import json
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluation.inception import InceptionScore
from data.dataset_params import get_dataset, get_collate_fn
from data.args import get_args, print_args, init_args
from models.meta_models import AG2VideoModel, MetaDiscriminatorModel
from models.metrics import jaccard
from models.spade_models.loss_model import LossModel
from models.utils import batch_to, log_scalar_dict, remove_dummies_and_padding
from models.spade_models.networks.sync_batchnorm import DataParallelWithCallback
import torch.distributed as dist
from models.vis import save_images

torch.backends.cudnn.benchmark = True


def restore_checkpoint(args, model, gans_model, discriminator, optimizer_graph, optimizer_gen, device):
    if args.checkpoint_name is None:
        raise Exception('You should pre-train the model on your training data first')

    # Load pre-trained weights for fine-tune
    checkpoint = torch.load(args.checkpoint_name, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.cuda()
    gans_model.load_state_dict(checkpoint['gans_model_state'])
    optimizer_graph.load_state_dict(checkpoint['optim_state_graph'])

    for state in optimizer_graph.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    optimizer_gen.load_state_dict(checkpoint['optim_state_gen'])

    for state in optimizer_gen.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    discriminator.img_discriminator.load_state_dict(checkpoint['d_img_state'])
    discriminator.optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

    # Load Epoch and Iteration num.
    t = checkpoint['counters']['t']
    epoch = checkpoint['counters']['epoch']

    return checkpoint["vocab"], epoch, t


def freeze_weights(model):
    print(" >> Freeze graph model weights:")
    if hasattr(model, 'acts_to_layout'):
        for param in model.acts_to_layout.parameters():
            param.requires_grad = False


def build_test_dsets(args):
    test_dset = get_dataset(args.dataset, 'test', args)
    vocab = test_dset.vocab
    collate_fn = get_collate_fn(args)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': False,
        'collate_fn': partial(collate_fn, vocab),
    }

    test_loader = DataLoader(test_dset, **loader_kwargs)
    return test_loader, test_dset.vocab


def build_train_val_loaders(args):
    train_dset = get_dataset(args.dataset, 'train', args)
    train_graph_dset = get_dataset(args.dataset, 'train_graph', args)
    val_dset = get_dataset(args.dataset, 'val', args)
    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))
    collate = get_collate_fn(args)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers // 2,
        'sampler': get_sampler(train_dset),
        'shuffle': True,
        'drop_last': True,
        'pin_memory': True,
        'collate_fn': partial(collate, vocab),
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    train_graph_loader = DataLoader(train_graph_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    loader_kwargs['sampler'] = get_sampler(val_dset)
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, train_graph_loader, val_loader


def build_test_loader(args):
    val_dset = get_dataset(args.dataset, 'test', args)
    vocab = json.loads(json.dumps(val_dset.vocab))
    collate = get_collate_fn(args)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers // 2,
        'sampler': get_sampler(val_dset),
        'shuffle': False,
        'drop_last': True,
        'pin_memory': True,
        'collate_fn': partial(collate, vocab),
    }
    loader_kwargs['sampler'] = get_sampler(val_dset)
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, val_loader


def get_sampler(dataset):
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    return sampler


def check_model(args, loader, model, gans_model, inception_score, use_gt=True, full_test=False):
    model.eval()
    num_samples = 0
    total_iou = 0.
    total_iou_05 = 0.
    total_iou_03 = 0.
    total_boxes = 0.
    last_samples = min(len(loader), args.num_val_samples)
    if inception_score is not None:
        inception_score.clean()

    image_df = {
        'video_id': [],
        'avg_iou': [],
        'iou03': [],
        'iou05': [],
        "predicted_boxes": [],
        "gt_boxes": [],
        "number_of_objects": []
    }
    samples = {}
    first_batch = 1
    with torch.no_grad():
        for batch in loader:
            batch = batch_to(batch)
            imgs, objs, boxes, triplets, actions, video_id = batch

            # Run the model as it has been run during training
            if use_gt:
                model_out = model(imgs, objs, triplets, actions, boxes_gt=boxes, test_mode=True, use_gt=True)
            else:
                model_out = model(imgs, objs, triplets, actions, boxes_gt=boxes[:, :1], test_mode=True,
                                  use_gt=False)
            imgs_pred, boxes_pred, flows_pred, conf_pred, _ = model_out

            if boxes_pred is not None:
                boxes_pred = torch.clamp(boxes_pred, 0., 1.)
            if imgs_pred is not None:
                bs, nt, ch, h, w = imgs_pred.size()
                inception_score(imgs_pred.contiguous().view(-1, ch, h, w))

            if not use_gt:
                image_df['video_id'].extend(video_id)

                for i in range(boxes.size(0)):
                    boxes_sample = boxes[i, 1:]
                    boxes_pred_sample = boxes_pred[i, 1:]
                    boxes_pred_sample, boxes_sample = \
                        remove_dummies_and_padding(boxes_sample, objs[i], args.vocab,
                                                   [boxes_pred_sample, boxes_sample])
                    iou, iou05, iou03 = jaccard(boxes_pred_sample, boxes_sample)
                    total_iou += iou.sum()
                    total_iou_05 += iou05.sum()
                    total_iou_03 += iou03.sum()
                    total_boxes += float(iou.shape[0])

                    image_df['avg_iou'].append(np.mean(iou))
                    image_df['iou03'].append(np.mean(iou03))
                    image_df['iou05'].append(np.mean(iou03))
                    image_df['predicted_boxes'].append(str(boxes_pred_sample.cpu().numpy().tolist()))
                    image_df['gt_boxes'].append(str(boxes_sample.cpu().numpy().tolist()))
                    image_df["number_of_objects"].append(len(objs[i]))

            num_samples += imgs.size(0)

            # Save last 10 videos results
            if first_batch:
                first_batch = 0

                # if not args.skip_graph_model
                if not samples:
                    samples['video_id'] = [video_id]
                    samples['vids'] = [imgs]
                    samples['gt_boxes'] = [boxes]
                    samples['pred_boxes'] = [boxes_pred]
                else:
                    samples['video_id'].append(video_id)
                    samples['vids'].append(imgs)
                    samples['gt_boxes'].append(boxes)
                    samples['pred_boxes'].append(boxes_pred)

                if use_gt:
                    if not 'pred_vids_gt_boxes' in samples:
                        samples['pred_vids_gt_boxes'] = [imgs_pred]
                    else:
                        samples['pred_vids_gt_boxes'].append(imgs_pred)
                    if not 'pred_vids_gt_boxes_boxes' in samples:
                        samples['pred_vids_gt_boxes_boxes'] = [boxes]
                    else:
                        samples['pred_vids_gt_boxes_boxes'].append(boxes)
                else:
                    if not 'pred_vids_pred_boxes' in samples:
                        samples['pred_vids_pred_boxes'] = [imgs_pred]
                    else:
                        samples['pred_vids_pred_boxes'].append(imgs_pred)
                    if not 'pred_vids_pred_boxes_boxes' in samples:
                        samples['pred_vids_pred_boxes_boxes'] = [boxes_pred]
                    else:
                        samples['pred_vids_pred_boxes_boxes'].append(boxes_pred)

            if not full_test and args.num_val_samples and num_samples >= args.num_val_samples:
                break

        mean_losses = {}
        if not use_gt:
            mean_losses.update({'avg_iou': total_iou / total_boxes,
                                'total_iou_05': total_iou_05 / total_boxes,
                                'total_iou_03': total_iou_03 / total_boxes})
            mean_losses.update({'inception_mean': 0.0})
            mean_losses.update({'inception_std': 0.0})

        inception_mean, inception_std = inception_score.compute_score(splits=5)
        mean_losses.update({'inception_mean': inception_mean})
        mean_losses.update({'inception_std': inception_std})

    model.train()
    return mean_losses, samples, pd.DataFrame.from_dict(image_df)

def check_model_iou(args, loader, model, gans_model, inception_score, use_gt=True, full_test=False):
    model.eval()
    num_samples = 0
    total_iou = 0.
    total_iou_05 = 0.
    total_iou_03 = 0.
    total_boxes = 0.
    last_samples = min(len(loader), args.num_val_samples)

    image_df = {
        'video_id': [],
        'avg_iou': [],
        'iou03': [],
        'iou05': [],
        "predicted_boxes": [],
        "gt_boxes": [],
        "number_of_objects": []
    }
    samples = {}
    first_batch = 1
    with torch.no_grad():
        for batch in loader:
            batch = batch_to(batch)
            imgs, objs, boxes, triplets, actions, video_id = batch

            # Run the model as it has been run during training
            if use_gt:
                model_out = model(imgs, objs, triplets, actions, boxes_gt=boxes, test_mode=True, use_gt=True)
            else:
                model_out = model(imgs, objs, triplets, actions, boxes_gt=boxes[:, :1], test_mode=True,
                                  use_gt=False)
            imgs_pred, boxes_pred, flows_pred, conf_pred, _ = model_out

            if boxes_pred is not None:
                boxes_pred = torch.clamp(boxes_pred, 0., 1.)

            image_df['video_id'].extend(video_id)

            for i in range(boxes.size(0)):
                boxes_sample = boxes[i, 1:]
                boxes_pred_sample = boxes_pred[i, 1:]
                boxes_pred_sample, boxes_sample = \
                    remove_dummies_and_padding(boxes_sample, objs[i], args.vocab,
                                               [boxes_pred_sample, boxes_sample])
                iou, iou05, iou03 = jaccard(boxes_pred_sample, boxes_sample)
                total_iou += iou.sum()
                total_iou_05 += iou05.sum()
                total_iou_03 += iou03.sum()
                total_boxes += float(iou.shape[0])

                image_df['avg_iou'].append(np.mean(iou))
                image_df['iou03'].append(np.mean(iou03))
                image_df['iou05'].append(np.mean(iou03))
                image_df['predicted_boxes'].append(str(boxes_pred_sample.cpu().numpy().tolist()))
                image_df['gt_boxes'].append(str(boxes_sample.cpu().numpy().tolist()))
                image_df["number_of_objects"].append(len(objs[i]))

            num_samples += imgs.size(0)

            # Save last 10 videos results
            if not full_test and args.num_val_samples and num_samples >= args.num_val_samples:
                break
    mean_losses = {}
    mean_losses.update({'avg_iou': total_iou / total_boxes,
                        'total_iou_05': total_iou_05 / total_boxes,
                        'total_iou_03': total_iou_03 / total_boxes})
    mean_losses.update({'inception_mean': 0.0})
    mean_losses.update({'inception_std': 0.0})

    print(mean_losses)
    model.train()

    return None, samples, pd.DataFrame.from_dict(image_df)

def cache_data(loader, num_workers, worker_id):
    ds = loader.dataset
    ds_size = len(ds) // num_workers
    for i in tqdm(range(worker_id * ds_size, (worker_id + 1) * ds_size)):
        ds[i]


def main(args):
    logger = logging.getLogger(__name__)
    args.vocab, train_loader, train_graph_loader, val_loader = build_train_val_loaders(args)

    if args.cache_data:
        cache_data(train_loader, args.num_workers, args.worker_id)
        cache_data(train_graph_loader, args.num_workers, args.worker_id)
        cache_data(val_loader, args.num_workers, args.worker_id)
        exit(0)

    init_args(args)
    learning_rate = args.learning_rate
    print_args(args)

    if not os.path.isdir(args.output_dir):
        print('Checkpoints directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)
    json.dump(vars(args), open(os.path.join(args.output_dir, 'run_args.json'), 'w'))
    writer = SummaryWriter(args.output_dir)
    float_dtype = torch.cuda.FloatTensor


    # setup device - CPU or GPU
    device = torch.device("cuda:{gpu}".format(gpu=args.gpu_ids[0]) if args.use_cuda else "cpu")
    print(" > Active GPU ids: {}".format(args.gpu_ids))
    print(" > Using device: {}".format(device.type))

    model = AG2VideoModel(args, device)
    model.type(float_dtype)

    optimizer_graph = torch.optim.Adam(model.acts_to_boxes.parameters(), lr=learning_rate,
                                       betas=(args.beta1, 0.999))
    optimizer_generator = torch.optim.Adam(list(set(model.parameters()) - set(model.acts_to_boxes.parameters())),
                                           lr=learning_rate, betas=(args.beta1, 0.999))

    print(model)

    discriminator = MetaDiscriminatorModel(args)
    print(discriminator)
    gans_model = LossModel(args, discriminator=discriminator)
    gans_model = DataParallelWithCallback(gans_model, device_ids=args.gpu_ids).to(device)

    epoch, t = 0, 0
    # Restore checkpoint
    if args.restore_checkpoint:
        args.vocab, epoch, t = restore_checkpoint(args, model, gans_model, discriminator, optimizer_graph, optimizer_generator,
                                      device)
    for m in [model, discriminator]:
        m.train()

    # Freeze graph weights
    if args.freeze_graph:
        freeze_weights(model)

    # Init Inception Score
    inception_score = InceptionScore(device, batch_size=args.batch_size, resize=True)
    # Run Epoch

    if not args.graph_only:
        train_loader_iter = iter(train_loader)
    train_graph_loader_iter = iter(train_graph_loader)
    G_losses = {}
    D_losses = {}
    while True:
        epoch += 1
        print('Starting epoch %d' % epoch)

        while t < args.num_iterations:
            # Save checkpoint
            if t % args.checkpoint_every == 0:
                checkpoint_path = os.path.join(args.output_dir, 'itr_%s.pt' % t)
                print('Saving checkpoint to ', checkpoint_path)
                save_checkpoint(args, checkpoint_path, discriminator, epoch, gans_model, model, optimizer_graph, optimizer_generator, t)

                # GT Boxes
                print('checking: input box as GT')
                gt_val_losses, gt_val_samples, _ = check_model(args, val_loader, model, gans_model,
                                                               inception_score, use_gt=True, full_test=False)
                log_scalar_dict(writer, gt_val_losses, 'use_gt/loss', t)
                log_results(gt_val_losses, t, prefix='GT VAL')
                save_images(args, t, gt_val_samples, dir_name='gt_val')

                # Pred Boxes
                val_losses, val_samples, _ = check_model(args, val_loader, model, gans_model,
                                                         inception_score, use_gt=False, full_test=False)
                log_scalar_dict(writer, val_losses, 'no_use_gt/loss', t)
                log_results(val_losses, t, prefix='VAL')
                save_images(args, t, val_samples, dir_name='val')


            # train end to end, on shorter sequence
            if not args.graph_only:

                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    batch = next(train_loader_iter)

                except Exception as e:
                    print("Error sample, continue.")
                    logger.exception(e)
                    continue

                try:
                    batch = batch_to(batch)
                    imgs, objs, boxes, triplets, actions, video_id = batch
                    if imgs is None:
                        continue

                    # Run segments per videos
                    model_out = model(imgs, objs, triplets, actions, boxes_gt=boxes, test_mode=False,
                                      use_gt=True)
                    G_losses = gans_model(batch, model_out, mode="compute_generator_loss")
                    G_losses = {k: v.mean() for k, v in G_losses.items()}
                    if (torch.isnan(G_losses['GAN_Img']) or (
                            G_losses.get('GAN_Feat', 0) and torch.isnan(G_losses['GAN_Feat']))):
                        print("Error: video_id {} is NAN!".format(video_id))
                        continue

                    log_scalar_dict(writer, G_losses, 'train/loss', t)

                    optimizer_generator.zero_grad()
                    G_losses["total_loss"].backward()
                    optimizer_generator.step()

                    D_losses = gans_model(batch, model_out, mode="compute_discriminator_loss")
                    D_losses = {k: v.mean() for k, v in D_losses.items()}
                    log_scalar_dict(writer, D_losses, 'train/loss', t)
                    set_optimizer_loss(D_losses, discriminator, args)

                except Exception as e:
                    print("Error in video_id: {}".format(video_id))
                    logger.exception(e)

            # train graph on long sequences
            G_graph_losses = {}
            try:
                batch = next(train_graph_loader_iter)
            except StopIteration:
                train_graph_loader_iter = iter(train_graph_loader)
                batch = next(train_graph_loader_iter)
            except Exception as e:
                print("Error sample, continue.")
                logger.exception(e)
                continue

            try:
                batch = batch_to(batch)
                imgs, objs, boxes, triplets, actions, video_id = batch
                model_out = model(imgs, objs, triplets, actions, boxes_gt=boxes, test_mode=False,
                                  graph_only=True)
                G_graph_losses = gans_model(batch, model_out, mode="compute_graph_loss")
                G_graph_losses = {k: v.mean() for k, v in G_graph_losses.items()}
                log_scalar_dict(writer, G_graph_losses, 'train/loss', t)

                optimizer_graph.zero_grad()
                G_graph_losses["total_loss"].backward()
                optimizer_graph.step()

            except Exception as e:
                print("Error in video_id: {}".format(video_id))
                logger.exception(e)

            # evaluation
            if t % args.print_every == 0:
                print('t = %d / %d' % (t, args.num_iterations))

                for name, val in G_graph_losses.items():
                    print(' Graph [%s]: %.4f' % (name, val))

                for name, val in G_losses.items():
                    print(' G [%s]: %.4f' % (name, val))

                for name, val in D_losses.items():
                    print(' D [%s]: %.4f' % (name, val))

            t+=1
        t = 0
def log_results(semi_val_losses, t, prefix=''):
    print('Iter: {}, '.format(t) + prefix + ' avg_iou: %.4f' % semi_val_losses.get('avg_iou', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' total_iou_03: %.4f' % semi_val_losses.get('total_iou_03', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' total_iou_05: %.4f' % semi_val_losses.get('total_iou_05', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' Inception mean: %.4f' % semi_val_losses.get('inception_mean', 0.0))
    print('Iter: {}, '.format(t) + prefix + ' Inception STD: %.4f' % semi_val_losses.get('inception_std', 0.0))


def set_optimizer_loss(D_losses, discriminator, args):
    discriminator.optimizer_d_img.zero_grad()
    D_losses["total_img_loss"].backward()
    discriminator.optimizer_d_img.step()


def save_checkpoint(args, checkpoint_path, discriminator, epoch, gans_model, model, optimizer_graph, optimizer_gen, t):
    checkpoint_dict = {
        'model_state': model.state_dict(),
        'gans_model_state': gans_model.state_dict(),
        'd_img_state': discriminator.img_discriminator.state_dict(),
        'd_img_optim_state': discriminator.optimizer_d_img.state_dict(),
        'optim_state_gen': optimizer_gen.state_dict(),
        'vocab': args.vocab,
        'counters': {
            't': t,
            'epoch': epoch,
        }
    }
    if optimizer_graph:
        checkpoint_dict.update({'optim_state_graph': optimizer_graph.state_dict()})
    torch.save(checkpoint_dict, checkpoint_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
