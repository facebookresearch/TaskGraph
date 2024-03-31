# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np

def IoU_class(frame_assignment, gt_assignment, sample=None):
    """calculate IoU as done in COIN"""

    if not isinstance(gt_assignment, np.ndarray):
        gt_assignment = gt_assignment.numpy()
    else:
        gt_assignment = np.array(gt_assignment)
    if not isinstance(frame_assignment, np.ndarray):
        frame_assignment = frame_assignment.numpy()
    else:
        frame_assignment = np.array(frame_assignment)

    if sample is not None:
        # color frame assignment with classes
        for i, cls_id in enumerate(sample["step_ids"]):
            frame_assignment[frame_assignment == i] = cls_id

    present_classes = np.unique(gt_assignment)
    present_classes = present_classes[present_classes > -1]
    intersection, union = 0, 0
    for s, cls_id in enumerate(present_classes):
        gt_cls_seg = gt_assignment == cls_id
        pred_cls_seg = frame_assignment == cls_id
        intersection += np.logical_and(gt_cls_seg, pred_cls_seg).sum()
        union += np.logical_or(gt_cls_seg, pred_cls_seg).sum()
    return intersection / union


def Acc_class(frame_assignment, gt_assignment, sample=None, use_negative=True):
    """calculate IoU as done in COIN"""
    # color frame assignment with classes
    if not isinstance(gt_assignment, np.ndarray):
        gt_assignment = gt_assignment.numpy()
    else:
        gt_assignment = np.array(gt_assignment)
    if not isinstance(frame_assignment, np.ndarray):
        frame_assignment = frame_assignment.numpy()
    else:
        frame_assignment = np.array(frame_assignment)

    if sample is not None:
        for i, cls_id in enumerate(sample["step_ids"]):
            frame_assignment[frame_assignment == i] = cls_id

    if not use_negative:
        frame_assignment = frame_assignment[gt_assignment > -1]
        gt_assignment = gt_assignment[gt_assignment > -1]
    return (frame_assignment == gt_assignment).astype(float).mean()