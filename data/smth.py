import json
import os
import pickle as pkl
import random
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from data.SomethingElse.config import action_to_number_of_instances, action_to_num_objects, valid_actions
from data.args import ALIGN_CORNERS
from models import group_transforms
from models.video_transforms import GroupMultiScaleCrop


class SmthElseDataset(Dataset):

    def __init__(self, data_root, is_test=False, is_val=False, debug=False, nframes=301,
                 image_size=(64, 64), fps=12, frames_per_action=16, initial_frames_per_sample=16,
                 max_samples=None, include_relationships=True, resize_or_crop='resize',
                 fine_size=64, load_size=64, aspect_ratio=1, no_flip=True, labels=None):

        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(SmthElseDataset, self).__init__()
        self.data_root = data_root
        self.videos_path = os.path.join(self.data_root, "videos")
        self.scenes_path = os.path.join(self.data_root, "scenes")
        self.lists_path = os.path.join(self.data_root, "lists")
        self.fps = fps
        self.nframes = nframes
        self.initial_frames_per_sample = initial_frames_per_sample  # before subsampling
        self.frames_per_action = frames_per_action
        self.resize_or_crop = resize_or_crop
        self.fine_size = fine_size
        self.load_size = load_size
        self.aspect_ratio = aspect_ratio
        self.no_flip = no_flip
        self.is_val = is_val
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        self.is_test = is_test
        
        # Get videos
        self.labels = pd.read_csv(labels)
        if "err" in self.labels.columns:
            self.labels = self.labels[pd.isnull(self.labels['err'])]


        # Get vocab mapping
        self.vocab = {}
        self.vocab["action_idx_to_name"] = action_to_number_of_instances

        # actions
        self.vocab["action_name_to_idx"] = {v: i for i, v in enumerate(self.vocab["action_idx_to_name"])}
        self.vocab['pred_name_to_idx'] = {'__in_image__': 0, 'right': 1, "above": 2, "below": 3, "left": 4,
                                          'surrounding': 5, 'inside': 6, 'cover': 7, '__padding__': 8}
        self.vocab['pred_idx_to_name'] = {v: k for k, v in self.vocab['pred_name_to_idx'].items()}

        # attributes
        self.vocab["attributes"] = {}
        self.vocab["reverse_attributes"] = {}
        # with open(os.path.join(self.data_root, 'offical_release_boxes/objects.pkl'), 'rb') as f:
        #     self.vocab["reverse_attributes"]['object'] = pkl.load(f)
        with open(os.path.join(self.data_root, 'offical_release_boxes/objs_mapping.json'), 'rb') as f:
            self.objs_mapping = json.load(f)
            self.vocab["reverse_attributes"]['object'] = ['__image__'] + sorted(list(set(self.objs_mapping.values())))
        self.vocab["attributes"]['object'] = {v: k for k, v in enumerate(self.vocab["reverse_attributes"]['object'])}
        self.vocab['object_idx_to_name'] = self.vocab["reverse_attributes"]['object']
        self.vocab['object_name_to_idx'] = self.vocab["attributes"]['object']

        # Sort actions
        self.labels = self.labels[self.labels['template'].isin(self.vocab["action_idx_to_name"])]
        self.labels = self.labels[[is_action_valid(row) for i, row in self.labels.iterrows()]]

        # Sort objects
        self.labels = self.labels.apply(lambda row: object_mapping_func(row, self.objs_mapping), axis=1)
        self.labels = self.labels[[is_object_valid(row) for i, row in self.labels.iterrows()]]

        self.vid_names = list(self.labels['id'])
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Transformations
        self.set_transforms(image_size)

    def set_transforms(self, image_size=(224, 224)):
        self.image_size = image_size
        self.normalize = group_transforms.GroupNormalize(self.img_mean, self.img_std)
        self.transforms = [group_transforms.GroupResize(image_size)]
        self.transforms += [
            group_transforms.ToTensor(),
            group_transforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = T.Compose(self.transforms)

    def __len__(self):
        return len(self.vid_names)

    def extract_triplets(self, boxes):
        F = boxes.size(0)
        O = boxes.size(1) - 1
        real_boxes = [i for i in range(O)]
        total_triplets = []
        for f in range(F):
            triplets = []
            for cur in real_boxes:
                choices = [obj for obj in real_boxes if obj != cur]
                if len(choices) == 0 or not self.include_relationships:
                    break
                other = random.choice(choices)
                if random.random() > 0.5:
                    s, o = cur, other
                else:
                    s, o = other, cur

                # Check for inside / surrounding
                sx0, sy0, sx1, sy1 = boxes[f][s]
                ox0, oy0, ox1, oy1 = boxes[f][o]
                sw = sx1 - sx0
                ow = ox1 - ox0
                sh = sy1 - sy0
                oh = oy1 - oy0
                mean_x = (sx0 + 0.5 * sw) - (ox0 + 0.5 * ow)
                mean_y = (sy0 + 0.5 * sh) - (oy0 + 0.5 * oh)
                theta = math.atan2(mean_y, mean_x)
                # d = obj_centers[s] - obj_centers[o]
                # theta = math.atan2(d[1], d[0])

                if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                    p = 'surrounding'
                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = 'inside'
                elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                    p = 'left'
                elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                    p = 'above'
                elif -math.pi / 4 <= theta < math.pi / 4:
                    p = 'right'
                elif math.pi / 4 <= theta < 3 * math.pi / 4:
                    p = 'below'
                p = self.vocab['pred_name_to_idx'][p]
                triplets.append([s, p, o])

            # Add dummy __in_image__ relationships for all objects
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            for i in range(O):
                triplets.append([i, in_image, O])
            total_triplets.append(triplets)

        total_triplets = torch.LongTensor(total_triplets)
        return total_triplets

    def extract_actions_split(self, boxes, num_objects, is_test):
        nr_instances = np.array([box['nr_instances'] for box in boxes])
        indices = np.where(nr_instances == num_objects)
        s_frame, e_frame = np.min(indices), np.max(indices) + 1
        f1 = s_frame.copy()
        if is_test:
            f1 = s_frame
            f2 = f1 + self.initial_frames_per_sample
        else:
            if e_frame - self.initial_frames_per_sample > s_frame:
                f1 = np.random.randint(s_frame, e_frame - self.initial_frames_per_sample)
            f2 = min(f1 + self.initial_frames_per_sample, e_frame)
        s = (f1 - s_frame + 1) / (e_frame - s_frame)
        e = (f2 - s_frame + 1) / (e_frame - s_frame)
        return f1, f2, s, e

    def extract_actions(self, objs, action_id, action_start, action_end):
        num_objs = len(objs["object"].cpu().numpy().astype('int'))
        hand_idx = num_objs - 1
        indices = objs["object"].cpu().numpy().astype('int')
        if self.vocab["object_idx_to_name"][indices[hand_idx]] != "hand":
            return False, "Last index is not hand"

        triplets = []
        prev = hand_idx
        for i in range(len(indices[:-1])):
            if self.vocab["object_idx_to_name"][indices[i]] == "hand":
                return False, "Multiple indices are hand"
            triplets.append([prev, action_id, i, action_start, action_end])
            prev = i

        if not len(triplets):
            return False, "No returned triplets"

        return True, torch.FloatTensor(triplets)

    def extract_bounding_boxes(self, boxes, img_shape, num_objects):
        """
        Get for each scene the bounding box
        :param scene: json data
        :param frames_id: list of frames ids
        :return: [F, O, 4]
        """

        object_indices = {}
        for timestep in boxes:
            for obj in timestep['labels']:
                obj_cat = (obj['standard_category'], obj['gt_annotation'], self.objs_mapping[obj['category']])
                if obj_cat not in object_indices:
                    object_indices[obj_cat] = len(object_indices)

        output_boxes = np.zeros((len(boxes), num_objects, 4))
        for i in range(len(boxes)):
            output_boxes[i] = output_boxes[i - 1]
            timestep = boxes[i]
            for obj in timestep['labels']:
                x1, x2, y1, y2 = obj['box2d']['x1'], obj['box2d']['x2'], obj['box2d']['y1'], obj['box2d']['y2']

                # Adding protection against bad boxes annotations
                if x1 == x2 and y1 == y2:
                    x1 = x2 = y1 = y2 = 0.0
                    print("Error: H=W=0 in {}".format(boxes[0]['name']))

                output_boxes[i, object_indices[(obj['standard_category'], obj['gt_annotation'],
                                                self.objs_mapping[obj['category']])]] = x1, y1, x2 - x1, y2 - y1
        reverse_object_indices = {v: k for k, v in object_indices.items()}
        objects = {"object": []}
        for i in range(len(reverse_object_indices)):
            objects["object"].append(self.vocab["object_name_to_idx"][reverse_object_indices[i][-1]])
        objects["object"] = torch.LongTensor(objects["object"])
        output_boxes = output_boxes / (img_shape * 2)
        if len(objects["object"]) != num_objects:
            return False, "len(objects) != num_objects", None
        return True, torch.FloatTensor(output_boxes), objects  # [x0, y0, w, h]

    def load_frames(self, frames_fns):
        return [Image.open(fn) for fn in frames_fns]

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triplets in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triplets: LongTensor of shape (T, 3) where triplets[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        try:
            # Choose video index
            video_id = self.vid_names[index]
            # Choose scene graph
            with open(os.path.join(self.data_root, f'offical_release_boxes/boxes_by_video_id/{video_id}/boxes.pkl'),
                      'rb') as f:
                boxes_metadata = pkl.load(f)
                boxes_metadata = clean_boxes_metadata(boxes_metadata)

            action_name = self.labels[self.labels['id'] == video_id].iloc[0]['template']
            action_idx = self.vocab["action_name_to_idx"][action_name]

            # Open video file
            imgs = self.extract_frames(boxes_metadata)
            if imgs is None:
                return False, "imgs is None"

            output = self.extract_actions_split(boxes_metadata, action_to_num_objects[action_name], self.is_test)
            if output is None:
                return False, "Mixed number of objects (occlusion?)"

            s_frame, e_frame, action_progress_s, action_progress_e = output
            chosen_video_id = f'{video_id}_{s_frame}-{e_frame}'
            thr_frames = self.initial_frames_per_sample if self.initial_frames_per_sample < 8 else 8
            if not self.is_val and (e_frame - s_frame) < thr_frames:
                return False, f"e_frame - s_frame < {thr_frames}"

            # Choose frames
            frames_lst = list(range(s_frame, e_frame))
            boxes_metadata = boxes_metadata[s_frame:e_frame]


            if self.is_test:
                frames_per_action = len(frames_lst)
                initial_frames_per_sample = len(frames_lst)
            else:
                frames_per_action = self.frames_per_action
                initial_frames_per_sample = self.initial_frames_per_sample

            frames_lst = frames_lst[0:initial_frames_per_sample: initial_frames_per_sample // frames_per_action]
            boxes_metadata = boxes_metadata[0:initial_frames_per_sample: initial_frames_per_sample // frames_per_action]
            initial_number_frames = len(frames_lst)
            padding = 0
            if len(frames_lst) < frames_per_action:
                padding = frames_per_action - initial_number_frames
                frames_lst = frames_lst + frames_lst[-1:] * padding
                boxes_metadata = boxes_metadata + boxes_metadata[-1:] * (frames_per_action - initial_number_frames)

            img_shape = self.load_frames(imgs[0:1])[0].size
            status, boxes, objs = self.extract_bounding_boxes(boxes_metadata, img_shape, action_to_num_objects[action_name])
            if not status:
                return False, status

            # Get actions - [A, 5]
            status, actions = self.extract_actions(objs, action_idx, action_progress_s, action_progress_e)
            if not status:
                return False, actions

            final_object_position = torch.zeros(actions.size(0), 2)
            actions = torch.cat([actions, final_object_position], dim=1)

            # Get triplets
            triplets = self.extract_triplets(boxes)

            try:
                frames = self.load_frames(imgs[frames_lst])
                vids = self.transforms(frames)
            except Exception as e:
                return False, "Error: Failed to load frames in video id: {}".format(video_id)

            return vids, objs, boxes, triplets, actions, chosen_video_id

        except Exception as e:
            return False, "Error: in video_id {} with {}".format(chosen_video_id, e)

    def extract_frames(self, boxes):
        paths = sorted([os.path.join(self.data_root, 'frames', box['name']) for box in boxes])
        return np.array(paths)


def is_action_valid(row):
    return action_to_num_objects[row['template']] == row['nr_instances'] and row['template'] in valid_actions


def object_mapping_func(row, objs_mapping):
    row['placeholders'] = [objs_mapping.get(obj, None) for obj in eval(row['placeholders'])]
    return row


def is_object_valid(row):
    return None not in row['placeholders']


def clean_boxes_metadata(boxes_metadata):
    """
    Get unique boxes metadata
    :param boxes_metadata:
    :return:
    """
    boxes_names = {b['name']: 0 for b in boxes_metadata}
    new_boxes_metadata = []
    for bb in boxes_metadata:
        if bb['name'] in boxes_names and boxes_names[bb['name']] == 0:
            boxes_names[bb['name']] += 1
            new_boxes_metadata.append(bb)
    return new_boxes_metadata
