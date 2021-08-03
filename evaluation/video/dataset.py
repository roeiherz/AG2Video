import cv2
import numpy as np
from torch.utils.data import Dataset
import skvideo
import torchvision.transforms as T

# Import ffmpeg from correct place
from evaluation.video.models.TSM.ops.transforms import ToTorchTensor, GroupNormalize
from skvideo.io import vreader


def scale(video, size):
    _, h, w, _ = video.shape

    if (w <= h and w == size) or (h <= w and h == size):
        return video
    if w < h:
        ow, oh = size, int(size * h / w)
    else:
        oh, ow = size, int(size * w / h)

    return np.stack([cv2.resize(img, (oh, ow)) for img in video])


def center_crop(video, crop_w, crop_h):
    t, h, w, c = video.shape
    start_w = w // 2 - crop_w // 2
    start_h = h // 2 - crop_h // 2

    return video[:, start_h: start_h + crop_h, start_w: start_w + crop_w, :]


def normalize(video, mean, std):
    return (video - mean) / std


def loop_padding(video, length):
    vlen = len(video)
    if vlen >= length:
        return video

    tile_shape = [1] * video.ndim
    tile_shape[0] = length // vlen + 1
    video = np.tile(video, tile_shape)
    video = video[:length]

    return video


def temporal_center_crop(video, length):
    if len(video) <= length:
        return video

    start = len(video) // 2 - length // 2

    return video[start: start + length]


class VideoDataset(Dataset):
    length = 16
    size = 112
    mean = [114.7748, 107.7354, 99.4750]
    std = [1, 1, 1]

    def __init__(self, root_path):
        self.video_paths = list(root_path.glob("*.mp4"))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, i):
        path = self.video_paths[i]

        # read video
        path_str = str(path)
        videogen = vreader(path_str)
        video = np.stack([frame for frame in videogen])
        video = video[..., ::-1]

        # spatial transforms
        video = scale(video, self.size)
        video = center_crop(video, self.size, self.size)
        video = normalize(video, self.mean, self.std)

        # temporal transforms
        video = temporal_center_crop(video, self.length)
        video = loop_padding(video, self.length)

        # (T, H, W, C) -> (C, T, H, W)
        video = video.transpose(3, 0, 1, 2)
        return video.astype(np.float32)


class VideoTSMDataset(Dataset):
    length = 16
    size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root_path):
        self.video_paths = list(root_path.glob("*.mp4"))
        self.transforms = [
            ToTorchTensor(),
            GroupNormalize(self.mean, self.std),
        ]
        self.transforms = T.Compose(self.transforms)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, i):
        path = self.video_paths[i]

        # read video
        path_str = str(path)
        videogen = vreader(path_str)
        video = np.stack([frame for frame in videogen])
        video = video[..., ::-1]

        # spatial transforms
        video = scale(video, self.size)
        video = center_crop(video, self.size, self.size)

        # temporal transforms
        video = temporal_center_crop(video, self.length)
        video = loop_padding(video, self.length)

        videos = self.transforms(video)  # (T, C, H, W)
        # # (T, C, H, W) -> (C, T, H, W)
        # videos = videos.permute(1, 0, 2, 3)
        return videos
