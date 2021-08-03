import torch
from data.cater import CATERDataset
from data.smth import SmthElseDataset
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


def collate_fn(vocab, batch):
    """
    Collate function to be used when wrapping a CATER in a DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triplets: FloatTensor of shape (T, 3) giving all triplets, where
    triplets[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triplets to images;
    triple_to_img[t] = n means that triplets[t] belongs to imgs[n].
    """
    all_vids, all_boxes, all_triplets, all_actions = [], [], [], []
    all_objs = []
    all_videos_ids = []

    max_triplets = 0
    max_objects = 0
    max_actions = 0

    batch = [item for item in batch if not isinstance(item[0], bool)]

    for i, (vid, objs, boxes, triplets, actions, video_id) in enumerate(batch):
        O = objs[list(objs.keys())[0]].size(0)
        T = triplets.size(1)  # triplets.size(2)
        A = actions.size(0)

        if max_objects < O:
            max_objects = O

        if max_triplets < T:
            max_triplets = T

        if max_actions < A:
            max_actions = A

    for i, (vid, objs, boxes, triplets, actions, video_id) in enumerate(batch):
        O = objs[list(objs.keys())[0]].size(0)
        A = actions.size(0)
        T = triplets.size(1)

        # Padded objs
        attributes = list(objs.keys())
        sorted(attributes)
        attributes_to_index = {attributes[i]: i for i in range(len(attributes))}
        attributes_objects = torch.zeros(len(attributes), max_objects, dtype=torch.long)

        for k, v in objs.items():
            # Padded objects
            if max_objects - O > 0:
                zeros_v = torch.zeros(max_objects - O, dtype=torch.long)
                padd_v = torch.cat([v, zeros_v])
            else:
                padd_v = v
            attributes_objects[attributes_to_index[k], :] = padd_v
        attributes_objects = attributes_objects.transpose(1, 0)

        # Padded boxes
        if max_objects - O > 0:
            padded_boxes = torch.FloatTensor([[-1, -1, -1, -1]]).repeat(boxes.size(0), max_objects - O, 1)
            boxes = torch.cat([boxes, padded_boxes], dim=1)

        # Padded triplets
        if max_triplets - T > 0:
            padded_triplets = torch.LongTensor([[0, vocab["pred_name_to_idx"]["__padding__"], 0]]).repeat(
                triplets.size(0), max_triplets - T, 1)
            triplets = torch.cat([triplets, padded_triplets], dim=1)

        # Padded actions
        if max_actions - A > 0:
            padded_actions = torch.FloatTensor([[0, vocab["action_name_to_idx"]["__padding__"], 0, 0, 0, 0, 0]]).repeat(
                max_actions - A, 1)
            actions = torch.cat([actions, padded_actions])

        all_vids.append(vid)
        all_videos_ids.append(video_id)
        all_objs.append(attributes_objects)
        all_boxes.append(boxes)
        all_triplets.append(triplets)
        all_actions.append(actions)

    try:
        all_vids = torch.stack(all_vids, dim=0)
        all_objs = torch.stack(all_objs, dim=0)
        all_boxes = torch.stack(all_boxes, dim=0)
        all_triplets = torch.stack(all_triplets, dim=0)
        all_actions = torch.stack(all_actions, dim=0)

        assert all_vids.size(0) == all_objs.size(0) == all_boxes.size(0) == all_triplets.size(0) == all_actions.size(0)
        out = (all_vids, all_objs, all_boxes, all_triplets, all_actions, all_videos_ids)
        return out

    except Exception as e:
        print("Exception in {}".format(e))
        return None, None, None, None, None, None


def get_dataset(name, partition, args):
    config = {
        "image_size": args.image_size,
        "resize_or_crop": args.resize_or_crop,
        "load_size": args.load_size,
        "fine_size": args.fine_size,
        "aspect_ratio": args.aspect_ratio,
        "no_flip": args.no_flip,
    }

    if name == 'smth_else':
        dataset_config = {
            "common": {
                "include_relationships": True,
                "max_samples": None,
                "data_root": os.path.join(dir_path, "SomethingElse"),
                "fps": 12,
                "debug": args.debug,
            },
            "train": {
                "labels": os.path.join(dir_path, "SomethingElse/train.csv"),
                "frames_per_action": args.frames_per_action,
                "initial_frames_per_sample": args.frames_per_action
            },
            "train_graph": {
                "labels": os.path.join(dir_path, "SomethingElse/train.csv"),
                "frames_per_action": 4 * args.frames_per_action_graph,
                "initial_frames_per_sample": 4 * args.frames_per_action_graph,
            },
            "val": {
                "labels": os.path.join(dir_path, "SomethingElse/val_split.csv"),
                "is_val": True,
                "frames_per_action": 16,
                "initial_frames_per_sample": 16,
            },
            "test": {
                "labels": os.path.join(dir_path, "SomethingElse/test_split.csv"),
                "is_val": True,
                "is_test": True,
                "frames_per_action": 16,
                "initial_frames_per_sample": 16,
            },
            "class": SmthElseDataset,
        }

    elif name == 'cater':
        dataset_config = {
            "common": {
                "include_relationships": True,
                "max_samples": None,
                "data_root": os.path.join(dir_path, "CATER/max2action"),
                "fps": 24,
                "debug": args.debug
            },
            "train": {
                "image_dir": os.path.join(dir_path, "CATER/train.txt"),
                "frames_per_action": args.frames_per_action,
                "initial_frames_per_sample": 3 * args.frames_per_action,
            },
            "train_graph": {
                "image_dir": os.path.join(dir_path, "CATER/train.txt"),
                "frames_per_action": 4 * args.frames_per_action_graph,
                "initial_frames_per_sample": 4 * 3 * args.frames_per_action_graph,
            },
            "val": {
                "image_dir": os.path.join(dir_path, 'CATER/val_split.txt'),
                "frames_per_action": 16,
                "initial_frames_per_sample": 16 * 3,
                "is_val": True,
            },
            "test": {
                "image_dir": os.path.join(dir_path, 'CATER/test_split.txt'),
                "is_val": True,
                "is_test": True,
                "frames_per_action": 16,
                "initial_frames_per_sample": 16 * 3,
            },
            "class": CATERDataset
        }
    else:
        raise ValueError("Wrong config.")

    config.update(dataset_config["common"])
    if args.debug:
        print("######### RUNNING IN DEBUG MODE, LOADING VALIDATION TO SAVE TIME #########")
        partition = 'val'
    config.update(dataset_config[partition])
    ds_instance = dataset_config['class'](**config)
    return ds_instance


def get_collate_fn(args):
    return collate_fn
