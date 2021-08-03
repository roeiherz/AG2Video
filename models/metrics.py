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


def xywh_to_points(points):
    bbox = points.clone()
    bbox[:, 2] = points[:, 0] + points[:, 2]
    bbox[:, 3] = points[:, 1] + points[:, 3]
    return bbox


def intersection(bbox_pred, bbox_gt):
    max_xy = torch.min(bbox_pred[:, 2:], bbox_gt[:, 2:])
    min_xy = torch.max(bbox_pred[:, :2], bbox_gt[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def jaccard(bbox_pred, bbox_gt):
    """
    :param bbox_pred: coordinates in xyhw
    :param bbox_gt: coordinates in xyhw
    :return: avg iou
    """

    bbox_pred = xywh_to_points(bbox_pred)
    bbox_gt = xywh_to_points(bbox_gt)

    inter = intersection(bbox_pred, bbox_gt)
    area_pred = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] - bbox_pred[:, 1])
    area_gt = (bbox_gt[:, 2] - bbox_gt[:, 0]) * (bbox_gt[:, 3] - bbox_gt[:, 1])
    union = area_pred + area_gt - inter
    iou = torch.div(inter, union)

    return iou.cpu().numpy(), (iou > 0.5).cpu().numpy(), (iou > 0.3).cpu().numpy()


def jaccard_masks(masks_pred, masks_gt):
    """
    :param masks_pred: [N, H, W]
    :param masks_gt: [N, H, W]
    :return: avg iou
    """

    masks_gt = masks_gt.float()
    masks_pred = masks_pred.float()

    inter = (masks_pred * masks_gt).sum()
    union = (masks_pred + masks_gt).sum() - inter
    iou = torch.div(inter, union)

    return iou.cpu().numpy()
