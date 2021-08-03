import os
import torch
from data.args import get_args, print_args, init_args
from models.graph_models.model import RuleBasedModel
from scripts.train import build_test_loader, check_model_iou

torch.backends.cudnn.benchmark = True


def main(args):
    args.vocab, val_loader = build_test_loader(args)

    init_args(args)
    print_args(args)

    if not os.path.isdir(args.output_dir):
        print('Checkpoints directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(args.output_dir)
    float_dtype = torch.cuda.FloatTensor

    # Define img_deprocess
    device = torch.device("cuda:{gpu}".format(gpu=args.gpu_ids[0]) if args.use_cuda else "cpu")
    print(" > Active GPU ids: {}".format(args.gpu_ids))
    print(" > Using device: {}".format(device.type))

    model = RuleBasedModel(args)
    model.type(float_dtype)

    gt_val_losses, gt_val_samples, _ = check_model_iou(args, val_loader, model, None, None, use_gt=True, full_test=False)


if __name__ == '__main__':
    args = get_args()
    main(args)
