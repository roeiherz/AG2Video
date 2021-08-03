#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

from models.metrics import xywh_to_points
from models.utils import remove_dummy_objects

"""
Functions for performing differentiable bilinear cropping of images, for use in
the object discriminator
"""


def crop_bbox_batch(imgs, objs, bbox, HH, WW=None, vocab=None, backend='cudnn'):
    """
    Inputs:
    - imgs: FloatTensor of shape (N, C, H, W)
    - bbox: FloatTensor of shape (B, 4) giving bounding box coordinates
    - bbox_to_feats: LongTensor of shape (B,) mapping boxes to feature maps;
      each element is in the range [0, N) and bbox_to_feats[b] = i means that
      bbox[b] will be cropped from feats[i].
    - HH, WW: Size of the output crops

    Returns:
    - crops: FloatTensor of shape (B, C, HH, WW) where crops[i] uses bbox[i] to
      crop from feats[bbox_to_feats[i]].
    """
    if backend == 'cudnn':
        return crop_bbox_batch_cudnn(imgs, objs, bbox, HH, WW, vocab=vocab)
    N, C, H, W = imgs.size()
    B = bbox.size(0)
    if WW is None: WW = HH
    dtype, device = imgs.dtype, imgs.device
    crops = torch.zeros(B, C, HH, WW, dtype=dtype, device=device)
    for i in range(N):
        mask = remove_dummy_objects(objs[i], vocab)
        cur_bbox = bbox[i][mask]
        n = cur_bbox.size(0)
        cur_feats = imgs[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_crops = crop_bbox(cur_feats, cur_bbox, HH, WW)
        crops[i] = cur_crops
    return crops


def _invperm(p):
    N = p.size(0)
    eye = torch.arange(0, N).type_as(p)
    pp = (eye[:, None] == p).nonzero()[:, 1]
    return pp


def crop_bbox_batch_cudnn(imgs, objs, bbox, HH, WW=None, vocab=None):
    B, N, C, H, W = imgs.size()
    if WW is None:
        WW = HH

    crops_b = []
    objs_b = []
    for b in range(B):
        feats_flat, bbox_flat, obj_flat = [], [], []
        mask = remove_dummy_objects(objs[b], vocab)
        for i in range(N):
            cur_bbox = bbox[b][i][mask]

            # Remove illegal boxes [0, 0, 0, 0]
            legal_boxes = (cur_bbox != 0).any(dim=-1)
            cur_bbox = cur_bbox[legal_boxes]
            curr_objs = objs[b][mask][legal_boxes].view(-1)
            n = cur_bbox.size(0)

            cur_feats = imgs[b][i].view(1, C, H, W).expand(n, C, H, W).contiguous()
            feats_flat.append(cur_feats)
            bbox_flat.append(cur_bbox)
            obj_flat.append(curr_objs)

        feats_flat_b = torch.cat(feats_flat, dim=0)
        bbox_flat_b = torch.cat(bbox_flat, dim=0)
        objs_flat_b = torch.cat(obj_flat, dim=0)

        crops = crop_bbox(feats_flat_b, bbox_flat_b, HH, WW, backend='cudnn')
        crops_b.append(crops)
        objs_b.append(objs_flat_b)

    return crops_b, objs_b


def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):
    """
    Take differentiable crops of feats specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, H, W)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - HH, WW: Size of the output crops.

    Returns:
    - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
      feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    bbox = xywh_to_points(bbox)
    N = feats.size(0)
    assert bbox.size(0) == N
    assert bbox.size(1) == 4
    if WW is None: WW = HH
    if backend == 'cudnn':
        # Change box from [0, 1] to [-1, 1] coordinate system
        bbox = 2 * bbox - 1
    x0, y0, x1, y1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
    Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
    if backend == 'jj':
        return bilinear_sample(feats, X, Y)
    elif backend == 'cudnn':
        grid = torch.stack([X, Y], dim=3)
        return F.grid_sample(feats, grid, align_corners=True)


def bilinear_sample(feats, X, Y):
    """
    Perform bilinear sampling on the features in feats using the sampling grid
    given by X and Y.

    Inputs:
    - feats: Tensor holding input feature map, of shape (N, C, H, W)
    - X, Y: Tensors holding x and y coordinates of the sampling
      grids; both have shape shape (N, HH, WW) and have elements in the range [0, 1].
    Returns:
    - out: Tensor of shape (B, C, HH, WW) where out[i] is computed
      by sampling from feats[idx[i]] using the sampling grid (X[i], Y[i]).
    """
    N, C, H, W = feats.size()
    assert X.size() == Y.size()
    assert X.size(0) == N
    _, HH, WW = X.size()

    X = X.mul(W)
    Y = Y.mul(H)

    # Get the x and y coordinates for the four samples
    x0 = X.floor().clamp(min=0, max=W - 1)
    x1 = (x0 + 1).clamp(min=0, max=W - 1)
    y0 = Y.floor().clamp(min=0, max=H - 1)
    y1 = (y0 + 1).clamp(min=0, max=H - 1)

    # In numpy we could do something like feats[i, :, y0, x0] to pull out
    # the elements of feats at coordinates y0 and x0, but PyTorch doesn't
    # yet support this style of indexing. Instead we have to use the gather
    # method, which only allows us to index along one dimension at a time;
    # therefore we will collapse the features (BB, C, H, W) into (BB, C, H * W)
    # and index along the last dimension. Below we generate linear indices into
    # the collapsed last dimension for each of the four combinations we need.
    y0x0_idx = (W * y0 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x0_idx = (W * y1 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y0x1_idx = (W * y0 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x1_idx = (W * y1 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)

    # Actually use gather to pull out the values from feats corresponding
    # to our four samples, then reshape them to (BB, C, HH, WW)
    feats_flat = feats.view(N, C, H * W)
    v1 = feats_flat.gather(2, y0x0_idx.long()).view(N, C, HH, WW)
    v2 = feats_flat.gather(2, y1x0_idx.long()).view(N, C, HH, WW)
    v3 = feats_flat.gather(2, y0x1_idx.long()).view(N, C, HH, WW)
    v4 = feats_flat.gather(2, y1x1_idx.long()).view(N, C, HH, WW)

    # Compute the weights for the four samples
    w1 = ((x1 - X) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w2 = ((x1 - X) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w3 = ((X - x0) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w4 = ((X - x0) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)

    # Multiply the samples by the weights to give our interpolated results.
    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return out


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


if __name__ == '__main__':
    import numpy as np
    from scipy.misc import imread, imsave, imresize

    cat = imresize(imread('cat.jpg'), (256, 256), anti_aliasing=True)
    dog = imresize(imread('dog.jpg'), (256, 256), anti_aliasing=True)
    feats = torch.stack([
        torch.from_numpy(cat.transpose(2, 0, 1).astype(np.float32)),
        torch.from_numpy(dog.transpose(2, 0, 1).astype(np.float32))],
        dim=0)

    boxes = torch.FloatTensor([
        [0, 0, 1, 1],
        [0.25, 0.25, 0.75, 0.75],
        [0, 0, 0.5, 0.5],
    ])

    box_to_feats = torch.LongTensor([1, 0, 1]).cuda()

    feats, boxes = feats.cuda(), boxes.cuda()
    crops = crop_bbox_batch_cudnn(feats, boxes, box_to_feats, 128)
    for i in range(crops.size(0)):
        crop_np = crops.data[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        imsave('out%d.png' % i, crop_np)
