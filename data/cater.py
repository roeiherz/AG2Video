import json, os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import cv2
from PIL import Image
from glob import glob
from data.args import ALIGN_CORNERS
from models.video_transforms import GroupMultiScaleCrop
from models import group_transforms
from skvideo.io import FFmpegReader


class CATERDataset(Dataset):

    def __init__(self, image_dir, data_root, is_test=False, is_val=False, debug=False, nframes=301, frames_mapping=None,
                 image_size=(64, 64), fps=24, frames_per_action=16, initial_frames_per_sample=48,
                 max_samples=None, include_relationships=True, resize_or_crop='resize', fine_size=64, load_size=64,
                 aspect_ratio=1, no_flip=True):

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
        super(CATERDataset, self).__init__()
        self.data_dir = image_dir
        self.data_root = data_root
        self.videos_path = os.path.join(self.data_root, "videos")
        self.scenes_path = os.path.join(self.data_root, "scenes")
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
        self.is_test = is_test
        self.max_samples = max_samples
        self.include_relationships = include_relationships
        # Get videos
        videos_folder = [vid.split('.')[0] for vid in os.listdir(self.videos_path)]
        # Get labels
        self.vid_labels = {}
        self.vid_names = []
        with open(self.data_dir, 'r') as f:
            for line in f:
                line_data = line.replace('\n', '').split(' ')
                name = line_data[0].split('.')[0]
                if name in videos_folder:
                    if name in ["CATER_new_004798", "CATER_new_006532", "CATER_new_001175", "CATER_new_000434",
                                "CATER_new_000346"]:
                        continue
                    self.vid_labels[name] = [int(n) for n in line_data[1].split(',')]
                    self.vid_names.append(name)

        # Get vocab mapping
        self.vocab = {}
        # self.vocab["use_object_embedding"] = False
        self.vocab['pred_name_to_idx'] = {'__in_image__': 0, 'right': 1, "above": 2, "below": 3, "left": 4,
                                          'surrounding': 5, 'inside': 6, '__padding__': 7}
        self.vocab['pred_idx_to_name'] = {v: k for k, v in self.vocab['pred_name_to_idx'].items()}
        self.vocab['action_name_to_idx'] = {'__in_image__': 0, '_no_op': 1, '_slide': 2, '_contain': 3, '_rotate': 4, '_pick_place': 5,
                                            '__padding__': 6}
        self.vocab['action_idx_to_name'] = {v: k for k, v in self.vocab['action_name_to_idx'].items()}

        # attributes
        self.vocab["attributes"] = {}
        self.vocab["attributes"]['shape'] = {'__image__': 0, 'cube': 1, 'sphere': 2, 'cylinder': 3, 'spl': 4, 'cone': 5}
        self.vocab["attributes"]["color"] = {'__image__': 0, 'gray': 1, 'red': 2, 'blue': 3, 'green': 4, 'brown': 5,
                                             'purple': 6, 'cyan': 7, 'yellow': 8, 'gold': 9}
        self.vocab["attributes"]["material"] = {'__image__': 0, 'rubber': 1, 'metal': 2}
        self.vocab["attributes"]["size"] = {'__image__': 0, 'small': 1, 'large': 2, 'medium': 3}

        self.vocab["reverse_attributes"] = {}
        for attr in self.vocab["attributes"].keys():
            self.vocab["reverse_attributes"][attr] = {v: k for k, v in self.vocab["attributes"][attr].items()}

        self.vocab['object_name_to_idx'] = {}
        ind = 0
        for attr in self.vocab["attributes"].keys():
            for attr_label in self.vocab["attributes"][attr].keys():
                if ind != 0:
                    keyy = "{}_{}".format(attr_label, ind)
                    self.vocab['object_name_to_idx'][keyy] = ind
                else:
                    # __image__
                    self.vocab['object_name_to_idx'][attr_label] = ind
                ind += 1

        # Get jsons
        jsons = [scene_path for scene_path in os.listdir(self.scenes_path)
                 if scene_path.split('.')[0] in self.vid_labels.keys()]
        self.json_data = {}
        for name in jsons:
            full_path = os.path.join(self.scenes_path, name)
            with open(full_path, 'r') as f:
                parsed_name = name.split('.')[0]
                sg = json.load(f)
                self.json_data[parsed_name] = sg


        # NOTE: Single channel mean/stev (unlike pytorch Imagenet)
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        # Transformations
        self.set_transforms(image_size)

    def set_transforms(self, image_size=(224, 224)):
        self.image_size = image_size
        self.transforms = [group_transforms.GroupResize(image_size)]
        self.transforms += [
            group_transforms.ToTensor(),
            group_transforms.GroupNormalize(self.img_mean, self.img_std),
        ]
        self.transforms = T.Compose(self.transforms)

    def __len__(self):
        if self.max_samples is None:
            return len(self.vid_labels)
        return min(len(self.vid_labels), self.max_samples)

    def extract_objs(self, sg):
        objs = {}
        for attr in self.vocab["attributes"].keys():
            attr_list = [self.vocab["attributes"][attr][obj[attr]] for obj in sg['objects']]
            attr_list.append(self.vocab["attributes"][attr]['__image__'])
            objs[attr] = torch.LongTensor(attr_list)
        return objs

    def extract_triplets(self, boxes):
        F = boxes.size(0)
        O = boxes.size(1) - 1
        real_boxes = [i for i in range(O)]
        total_triplets = []
        for f in range(F):
            triplets = []
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            for i in range(O):
                triplets.append([i, in_image, O])
            total_triplets.append(triplets)

        total_triplets = torch.LongTensor(total_triplets)
        return total_triplets

    def extract_actions_split(self, sg, max_frame, is_test):
        obj_name_to_ind = {obj['instance']: id for id, obj in enumerate(sg['objects'])}
        actions = []
        start_frames = []
        end_frames = []
        for o1_name, data in sg['movements'].items():
            o1_id = obj_name_to_ind[o1_name]
            for d in data:
                action = d[0]
                o2_name = d[1]
                frame_s = d[2]
                frame_t = d[3]

                # Skip fewer frames
                if frame_t - frame_s < 12:
                    continue

                start_frames.append(frame_s)
                end_frames.append(frame_t)
                action_id = self.vocab['action_name_to_idx'][action]
                o2_id = obj_name_to_ind[o2_name] if o2_name is not None else o1_id
                actions.append([o1_id, action_id, o2_id, frame_s, frame_t])
        if is_test:
            start_a = min(start_frames)
            end_a = min(max(end_frames), start_a + self.initial_frames_per_sample)
        else:
            start_a = np.random.randint(0, min(max(end_frames), max_frame) - self.initial_frames_per_sample + 1)
            end_a = start_a + self.initial_frames_per_sample
        chosen_actions = torch.LongTensor([action for action in actions if not(action[3] > end_a or action[4] < start_a)]) # at least 1 frame overlap
        return chosen_actions, [start_a, end_a]

    def extract_actions(self, sg):
        obj_name_to_ind = {obj['instance']: id for id, obj in enumerate(sg['objects'])}
        actions = []
        start_frames = []
        end_frames = []
        temporal_tuples = []
        for o1_name, data in sg['movements'].items():
            o1_id = obj_name_to_ind[o1_name]
            for d in data:
                action = d[0]
                o2_name = d[1]
                frame_s = d[2]
                frame_t = d[3]

                # Skip fewer frames
                if frame_t - frame_s < 12:
                    continue

                temporal_tuples.append((frame_s, frame_t))
                start_frames.append(frame_s)
                end_frames.append(frame_t)
                action_id = self.vocab['action_name_to_idx'][action]
                o2_id = obj_name_to_ind[o2_name] if o2_name is not None else o1_id
                actions.append([o1_id, action_id, o2_id, frame_s, frame_t])

        # Choose a fixed number of actions
        actions = torch.LongTensor(actions)
        return actions

    def extract_bounding_boxes(self, scene):
        """
        Get for each scene the bounding box
        :param scene: json data
        :param frames_id: list of frames ids
        :return: [F, O, 4]
        """
        objs = scene['objects']

        boxes = []
        W = 320
        H = 240
        for i, obj in enumerate(objs):
            locations = np.array([v for k, v in list(obj['locations'].items())])
            points2d = self._project_3d_point(locations)
            cx = points2d[:, 0]
            cy = points2d[:, 1]
            cx = (cx + 1) * W / 2
            cy = (cy + 1) * H / 2

            if obj['shape'] == 'spl':
                if obj['size'] == 'large':
                    w_box = 35
                    h_box_s = h_box_l = 35
                if obj['size'] == 'medium':
                    w_box = 25
                    h_box_s = h_box_l = 25
                if obj['size'] == 'small':
                    w_box = 15
                    h_box_s = h_box_l = 15

            if obj['shape'] == 'cylinder':
                if obj['size'] == 'large':
                    w_box = 35
                    h_box_s = h_box_l = 35
                if obj['size'] == 'medium':
                    w_box = 25
                    h_box_s = h_box_l = 25
                if obj['size'] == 'small':
                    w_box = 15
                    h_box_s = h_box_l = 15

            if obj['shape'] == 'cone':
                if obj['size'] == 'large':
                    w_box = 35
                    h_box_s = 25
                    h_box_l = 40
                if obj['size'] == 'medium':
                    w_box = 25
                    h_box_s = 15
                    h_box_l = 30
                if obj['size'] == 'small':
                    w_box = 20
                    h_box_s = h_box_l = 20

            if obj['shape'] == 'sphere':
                if obj['size'] == 'large':
                    w_box = 35
                    h_box_s = 25
                    h_box_l = 40
                if obj['size'] == 'medium':
                    w_box = 25
                    h_box_s = h_box_l = 25
                if obj['size'] == 'small':
                    w_box = 15
                    h_box_s = h_box_l = 15

            if obj['shape'] == 'cube':
                if obj['size'] == 'large':
                    w_box = 35
                    h_box_s = h_box_l = 35
                if obj['size'] == 'medium':
                    w_box = 25
                    h_box_s = h_box_l = 25
                if obj['size'] == 'small':
                    w_box = 15
                    h_box_s = h_box_l = 15

            x_min_coord = cx - w_box
            y_min_coord = cy - h_box_s
            x_max_coord = cx + w_box
            y_max_coord = cy + h_box_l

            # boxes - [x0, y0, w, h]; normalized boxes 0-1
            boxes.append(np.transpose(
                [x_min_coord / W, y_min_coord / H, (x_max_coord - x_min_coord) / W, (y_max_coord - y_min_coord) / H]))

        boxes.append(np.tile([[0., 0., 1., 1.]], [x_min_coord.size, 1]))  # Add image coordinates
        boxes = np.transpose(boxes, (1, 0, 2))
        boxes = torch.FloatTensor(boxes)  # [x0, y0, w, h]
        return boxes

    def _project_3d_point(self, pts):
        """
        Args:
            pts: Nx3 matrix, with the 3D coordinates of the points to convert
        Returns:
            Nx2 matrix, with the coordinates of the point in 2D, from +1 to -1 for each dimension.
            The top left corner is the -1, -1
        """

        # This cam was extracted from the image_generation/render_videos code, for the camera fixed case
        CATER_CAM = [
            (1.4503, 1.6376, 0.0000, -0.0251),
            (-1.0346, 0.9163, 2.5685, 0.0095),
            (-0.6606, 0.5850, -0.4748, 10.5666),
            (-0.6592, 0.5839, -0.4738, 10.7452)]

        p = np.matmul(
            np.array(CATER_CAM),
            np.hstack((pts, np.ones((pts.shape[0], 1)))).transpose()).transpose()
        # The predictions are -1 to 1, Negating the 2nd to put low Y axis on top
        p[:, 0] /= p[:, -1]
        p[:, 1] /= -p[:, -1]
        return p[:, :2]

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

        # Choose video index
        video_id = self.vid_names[index]
        # Choose scene graph
        sg = self.json_data[video_id]

        # Open video file
        imgs = self.extract_frames(video_id)
        if imgs is None:
            return None, None, None, None, None, None

        actions, chosen_frame = self.extract_actions_split(sg, len(imgs) - 1, self.is_test)
        s_frame = chosen_frame[0]  # Start frame
        e_frame = chosen_frame[1]  # End frame
        frames_lst = list(range(s_frame, e_frame))

        frames_per_action = self.frames_per_action
        initial_frames_per_sample = self.initial_frames_per_sample

        assert len(frames_lst) == initial_frames_per_sample, "different size"
        frames_lst = frames_lst[0:initial_frames_per_sample: initial_frames_per_sample // frames_per_action]

        try:
            frames = self.load_frames(imgs[frames_lst])
            vids = self.transforms(frames)
        except Exception as e:
            print("Error: Failed to load frames in video id: {}".format(video_id))
            return None, None, None, None, None, None

        # Get boxes - [F, O, 4]
        all_boxes = self.extract_bounding_boxes(sg)
        boxes = all_boxes[frames_lst]

        # Get triplets - [F, T, 3]
        triplets = self.extract_triplets(boxes)
        # Get objects
        objs = self.extract_objs(sg)
        # Get actions - [A, 5]

        norm_actions = self.normalized_actions(actions, all_boxes, s_frame, e_frame)  # [A, 5]
        chosen_video_id = f'{video_id}_{s_frame}-{e_frame}'
        return vids, objs, boxes, triplets, norm_actions, chosen_video_id

    def extract_frames(self, video_id):

        video_cache_path = os.path.join(self.videos_path, video_id)
        if not os.path.exists(video_cache_path):
            try:
                os.makedirs(video_cache_path)
                video_path = os.path.join(self.videos_path, "{}.avi".format(video_id))
                reader = FFmpegReader(video_path, inputdict={},
                                      outputdict={"-r": "%d" % self.fps, "-vframes": "%d" % self.nframes})
                i = 0
                for img in reader.nextFrame():
                    cached_video_path = os.path.join(video_cache_path, f"{i:05}.png")
                    Image.fromarray(img).save(cached_video_path)
                    i += 1
            except Exception as e:
                print(e)
                return None

        imgs = sorted(glob(os.path.join(video_cache_path, "*.png")))
        if len(imgs) != self.nframes:
            print(f"Number of frames in {video_id} is {len(imgs)}")
            return None

        return np.array(imgs)

    def normalized_actions(self, actions, boxes, s_frame, e_frame):
        """
        Normalized the frames of the actions in range [0,16]
        :param actions: Tensor of Actions [A, 5]
        :param frames_range: [min_frame, max_frame]
        :return: Tensor of Actions [A, 5]
        """
        _, _, _, f1, f2 = actions.chunk(5, dim=-1)  # [A, 1]
        f1, f2 = [x.squeeze(-1).type(torch.float32) for x in [f1, f2]]  # [A, ]
        norm_t1 = (s_frame - f1) / (f2 - f1 + 1)
        norm_t2 = (e_frame - f1) / (f2 - f1 + 1)
        temporal = torch.stack([norm_t1, norm_t2], dim=-1)
        norm_actions = torch.cat([actions[:, :3].type(torch.FloatTensor), temporal], dim=-1)  # [A, 5]
        include_action = ~((norm_t1 > 1) | (norm_t2 < 0))
        norm_actions = norm_actions[include_action]

        final_object_position = boxes[f2[include_action].type(torch.LongTensor), norm_actions[:, 0].type(torch.LongTensor)][:, :2]
        add_gt = (norm_actions[:, 1] == self.vocab['action_name_to_idx']['_pick_place']) | (
                norm_actions[:, 1] == self.vocab['action_name_to_idx']['_slide'])
        final_object_position[~add_gt] = 0.
        return torch.cat([norm_actions, final_object_position], dim=1)

