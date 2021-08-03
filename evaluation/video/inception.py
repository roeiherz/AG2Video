import multiprocessing
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import requests
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import VideoDataset, VideoTSMDataset
from .models import resnext
import os
from .models.TSM.ops.models import TSN


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def prepare_inception_model(device: torch.device = torch.device("cpu"), weight_dir: str = ''):
    filename = "resnext-101-kinetics-ucf101_split1.pth"
    weight_path = '{}/{}'.format(weight_dir, filename)
    if not os.path.exists(weight_path):
        print(">> download trained model...")
        file_id = "1DmI6QBrh7xhme0jOL-3nEutJzesHZTqp"
        gdd.download_file_from_google_drive(file_id=file_id, dest_path=weight_path)

    # model = resnext.resnet101(num_classes=101, sample_size=112, sample_duration=16)
    model = resnext.resnet101(num_classes=101, sample_size=256, sample_duration=16)

    model_data = torch.load(str(weight_path), map_location="cpu")
    fixed_model_data = OrderedDict()
    for key, value in model_data["state_dict"].items():
        new_key = key.replace("module.", "")
        fixed_model_data[new_key] = value

    model.load_state_dict(fixed_model_data)
    model = model.to(device)
    model.eval()

    return model


def prepare_tsm_model(device: torch.device = torch.device("cpu"), weight_dir: str = ''):
    filename = "TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth"
    weight_path = '{}/{}'.format(weight_dir, filename)
    if not os.path.exists(weight_path):
        print("Error: TSM weights in {} are not exits!".format(weight_path))

    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weight_path)
    model = TSN(174, 16 if is_shift else 1, 'RGB',
                base_model='resnet50',
                consensus_type='avg',
                img_feature_dim=256,
                pretrain='imagenet',
                is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
                non_local='_nl' in weight_path,
                print_spec=False
                )

    checkpoint = torch.load(weight_path)
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    # replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
    #                 'base_model.classifier.bias': 'new_fc.bias',
    #                 }
    # for k, v in replace_dict.items():
    #     if k in base_dict:
    #         base_dict[v] = base_dict.pop(k)
    model.load_state_dict(base_dict)
    model = model.to(device)
    model.eval()

    return model


def forward_videos(model, dataloader, device, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    softmax = torch.nn.Softmax(dim=1)
    features, probs = [], []
    with torch.no_grad():
        for videos in tqdm(iter(dataloader), disable=not verbose):
            # foward samples
            videos = videos.to(device)
            _features, _probs = model(videos)

            # to cpu
            _features = _features.cpu().numpy()
            _probs = softmax(_probs).cpu().numpy()

            # add results
            features.append(_features)
            probs.append(_probs)

    return np.concatenate(features, axis=0), np.concatenate(probs, axis=0)


def create_conv_features(
        videos_path: Path, batchsize: int = 10, verbose: bool = False, weight_path: str = '', model: str = '') -> Tuple[
    np.ndarray, np.ndarray]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # init model and load pretrained weights
    if model == 'TSM':
        model = prepare_tsm_model(device, weight_path)
        dataset = VideoTSMDataset(videos_path)

    # load generated samples as pytorch dataset
    if model == 'ResNext101':
        model = prepare_inception_model(device, weight_path)
        dataset = VideoDataset(videos_path)

    nprocs = multiprocessing.cpu_count()
    if verbose:
        print(f">> found {len(dataset)} samples.")
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=0, pin_memory=True)

    # forward samples to the model and obtain results
    if verbose:
        print(f">> converting videos into conv features using inception model (on {device})...")

    try:
        features, probs = forward_videos(model, dataloader, device, verbose)
    except Exception as e:
        features, probs = None, None
        print("Error in {}".format(e))

    del model

    return features, probs
