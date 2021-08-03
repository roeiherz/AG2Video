import argparse
import os
import json
import random
from argparse import Namespace
import numpy as np

import cv2
import imageio
import torch
import pickle as pkl
from tqdm import tqdm
from models.meta_models import AG2VideoModel
from models.utils import batch_to
from models.vis import plot_vid
from scripts.train import build_test_loader

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--gpu_ids', type=int)
parser.add_argument('--save_test', type=int, default=1)
parser.add_argument('--save_actions', type=int, default=1)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--use_test', type=int, default=1)


def sample_to_cuda(s):
    return [t.cuda() if isinstance(t, torch.Tensor) else t for t in s]


def save_gif(image_list, fn, bgr2rgb=True, start_indication=False):
    if bgr2rgb:
        image_list = [cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB) for image in image_list]

    if start_indication:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 255, 255)
        thickness = 4
        blank = np.zeros(image_list[0].shape, dtype='uint8')
        image = [cv2.putText(blank, 'Start', org, font, fontScale, color, thickness, cv2.LINE_AA)] * 4
        image_list = image + image_list

    imageio.mimsave(fn, image_list)


def diagonal(objs, vids, boxes):
    norm_actions = [[1, 1, 0, 0, 1.01, 0, 0],
                    [1, 2, 0, 0, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def diagonal_down_left(objs, vids, boxes):
    norm_actions = [[1, 3, 0, 0, 1.01, 0, 0],
                    [1, 4, 0, 0, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def down(objs, vids, boxes):
    norm_actions = [[1, 3, 0, 0, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def left(objs, vids, boxes):
    norm_actions = [[1, 4, 0, 0, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def rightup_then_up(objs, vids, boxes):
    norm_actions = [[1, 2, 0, 0, 1.01, 0, 0],
                    [1, 1, 0, 0, 1.01, 0, 0],
                    [1, 1, 0, -0.3, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def right_then_up(objs, vids, boxes):
    norm_actions = [[1, 2, 0, 0, 1.01, 0, 0],
                    [1, 1, 0, 0, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def up(objs, vids, boxes):
    norm_actions = [[1, 1, 0, 0, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def right(objs, vids, boxes):
    norm_actions = [[1, 2, 0, 0, 1.01, 0, 0]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def converge(objs, vids, boxes):
    initial_img = vids[:, :1]
    num_objs = objs.shape[1] - 1
    # target = boxes[0, 0, 0, :2]
    norm_actions = []  # [[0, 3, 0, 0., 0.5, 0,0]]#float(target[0]), float(target[1])]]
    for j in range(1, num_objs):
        start = (float(j) / num_objs)
        norm_actions.append([j, 3, 0, start, 1.05, 0, 0])
        # norm_actions.append([j, 3, j-1, 0., 1., float(target[0]), float(target[1])])
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def converge_before(objs, vids, boxes):
    num_objs = objs.shape[1] - 1
    norm_actions = []  # [[0, 3, 0, 0., 0.5, 0,0]]#float(target[0]), float(target[1])]]
    for j in range(1, num_objs):
        start = -1 * (float(j) / num_objs)
        norm_actions.append([j, 3, 0, start, 2., 0, 0])
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def converge_after(objs, vids, boxes):
    initial_img = vids[:, :1]
    num_objs = objs.shape[1] - 1
    # target = boxes[0, 0, 0, :2]
    norm_actions = []  # [[0, 3, 0, 0., 0.5, 0,0]]#float(target[0]), float(target[1])]]
    for j in range(1, num_objs):
        start = -1 * (float(j) / num_objs) - 1
        norm_actions.append([j, 3, 0, start, 1.05, 0, 0])
        # norm_actions.append([j, 3, j-1, 0., 1., float(target[0]), float(target[1])])
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


def swap(objs, vids, boxes):
    num_objs = objs.shape[1] - 1
    obj_indices = list(range(num_objs))
    random.shuffle(obj_indices)
    index1, index2 = obj_indices[0], obj_indices[1]
    target1 = boxes[0, 0, index2]
    target2 = boxes[0, 0, index1]
    norm_actions = [[index1, 5, index1, -0.3, 1.3, float(target1[0]), float(target1[1])],
                    [index2, 2, index2, -0.3, 1.3, float(target2[0]), float(target2[1])]]
    norm_actions = torch.FloatTensor(norm_actions).unsqueeze(0)
    return norm_actions


actions_to_execute_cater = [
    {"action_name": "converge", "action_func": converge},
    {"action_name": "swap", "action_func": swap},
]

actions_to_execute_smth = [
    {"action_name": "down_left", "action_func": diagonal_down_left},
    {"action_name": "down", "action_func": down},
    {"action_name": "left", "action_func": left},
    {"action_name": "right", "action_func": right},
    {"action_name": "up", "action_func": up},
    {"action_name": "right_up", "action_func": diagonal},
]

actions_to_execute_dict = {
    "cater": actions_to_execute_cater,
    "smth_else": actions_to_execute_smth
}


def main():
    args = parser.parse_args()
    checkpoint = args.checkpoint
    train_args = json.load(open(os.path.join(os.path.dirname(checkpoint), 'run_args.json'), 'rb'))
    train_args["gpu_ids"] = [args.gpu_ids]
    train_args["batch_size"] = 1
    train_args["loader_num_workers"] = 0
    train_args["frames_per_action_graph"] = 1

    train_args_namespace = Namespace(**train_args)
    train_args_namespace.model_name = args.checkpoint.split('/')[-2]

    vocab, val_loader = build_test_loader(train_args_namespace)

    map_location = torch.device('cuda:%s' % train_args["gpu_ids"][0])
    print("loading: %s" % checkpoint)

    dir_p = os.path.join(args.output_dir, f"results_{train_args_namespace.dataset}", train_args_namespace.model_name)
    if args.use_test:
        dir_p += "_gt_layout"

    checkpoint = torch.load(checkpoint, map_location=map_location)
    model = AG2VideoModel(train_args_namespace, map_location)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    model.to(map_location)
    os.makedirs(dir_p, exist_ok=True)
    print(f"saving to: {dir_p}")
    val_loader = iter(val_loader)

    for _ in tqdm(range(len(val_loader)), total=len(val_loader)):
        try:
            b = next(val_loader)
            vids, objs, boxes, triplets, norm_actions, chosen_video_id = batch_to(b)

            if args.save_actions:
                actions_to_execute = actions_to_execute_dict[train_args_namespace.dataset]
                actions_in_video = norm_actions[0, :, 1].cpu().numpy()
                actions_in_video_indices = list(range(len(actions_in_video)))
                np.random.shuffle(actions_in_video_indices)

                actions_chosen = int(actions_in_video[actions_in_video_indices[0]])
                if train_args["dataset"] == "cater":
                    actions_chosen = str(actions_chosen)
                action_name = vocab["action_idx_to_name"][actions_chosen]
                cloned_norm_actions = norm_actions.clone()
                cloned_norm_actions[:, :, 3] = 0.
                cloned_norm_actions[:, :, 4] = 1.1
                if train_args["dataset"] == "cater":
                    initial_action = {"action_name": action_name, "action_func": lambda x, y, z: cloned_norm_actions[:,
                                                                                                 actions_in_video_indices[
                                                                                                     0]:
                                                                                                 actions_in_video_indices[
                                                                                                     0] + 1]}
                else:
                    initial_action = {"action_name": action_name, "action_func": lambda x, y, z: cloned_norm_actions}

                batch_actions_to_execute = actions_to_execute + [initial_action]
                # batch_actions_to_execute = [act for act in batch_actions_to_execute if act["action_name"] == "converge"]
                for act in batch_actions_to_execute:
                    first_obj = boxes[0][0][0].cpu().numpy()
                    if act["action_name"] == "down_left" and (first_obj[0] < 0.35 or first_obj[1] > 0.65):
                        continue
                    elif act["action_name"] == "right_up" and (first_obj[0] > 0.65 or first_obj[1] < 0.35):
                        continue
                    else:
                        pass

                    print(act["action_name"])
                    norm_actions = act["action_func"](objs, vids, boxes)
                    with torch.no_grad():
                        imgs_pred, boxes_pred, _, _, _ = model(vids, objs, triplets, norm_actions, boxes_gt=boxes,
                                                               test_mode=True, use_gt=False)

                    fn = os.path.join(dir_p, f"action_accuracy/{act['action_name']}/{chosen_video_id[0]}.gif")
                    os.makedirs(os.path.dirname(fn), exist_ok=True)
                    save_gif(plot_vid(imgs_pred[0]), fn, bgr2rgb=False, start_indication=True)

                    # timing - before
                    if act["action_name"] != "converge":
                        norm_actions[:, :, 3] = 0.
                        norm_actions[:, :, 4] = 3.
                    else:
                        norm_actions = converge_before(objs, vids, boxes)

                    with torch.no_grad():
                        imgs_pred, boxes_pred, _, _, _ = model(vids, objs, triplets, norm_actions,
                                                               boxes_gt=boxes, test_mode=True, use_gt=False)
                    fn = os.path.join(dir_p, f"action_timing/{act['action_name']}/before/{chosen_video_id[0]}.gif")
                    os.makedirs(os.path.dirname(fn), exist_ok=True)
                    save_gif(plot_vid(imgs_pred[0]), fn, bgr2rgb=False, start_indication=True)

                    # timing - after
                    if act["action_name"] != "converge":
                        norm_actions[:, :, 3] = -2
                        norm_actions[:, :, 4] = 1.
                    else:
                        norm_actions = converge_after(objs, vids, boxes)

                    with torch.no_grad():
                        imgs_pred, boxes_pred, _, _, _ = model(vids, objs, triplets, norm_actions,
                                                               boxes_gt=boxes, test_mode=True, use_gt=False)

                    fn = os.path.join(dir_p, f"action_timing/{act['action_name']}/after/{chosen_video_id[0]}.gif")
                    os.makedirs(os.path.dirname(fn), exist_ok=True)
                    save_gif(plot_vid(imgs_pred[0]), fn, bgr2rgb=False, start_indication=True)

                    fn = os.path.join(dir_p, f"gt_action/{act['action_name']}/{chosen_video_id[0]}.gif")
                    os.makedirs(os.path.dirname(fn), exist_ok=True)
                    save_gif(plot_vid(vids[0]), fn, bgr2rgb=False, start_indication=True)

            if args.save_test:
                with torch.no_grad():
                    print(triplets.shape)
                    imgs_pred, boxes_pred, _, _, _ = model(vids, objs, triplets, norm_actions, boxes_gt=boxes,
                                                           test_mode=True, use_gt=args.use_test)

                if boxes_pred is not None:
                    boxes_pred = boxes_pred.squeeze()
                    if boxes_pred.shape[-2] > 3:
                        boxes_pred = boxes_pred[:, :-1]
                perc_image = {'image': imgs_pred.squeeze(), 'box': boxes_pred}
                save_p = os.path.join(dir_p, "test")
                os.makedirs(save_p, exist_ok=True)
                fp_gif = os.path.join(save_p, chosen_video_id[0] + '.gif')
                fp_pkl = os.path.join(save_p, chosen_video_id[0] + '.pkl')
                transform_bgr = False  # train_args_namespace.dataset == 'cater'
                save_gif(plot_vid(imgs_pred[0]), fp_gif, bgr2rgb=transform_bgr)
                with open(fp_pkl, 'wb') as f:
                    pkl.dump(perc_image, f)
        except Exception as e:
            print(chosen_video_id, e)


if __name__ == "__main__":
    main()
