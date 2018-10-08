# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np

def yolo_boxes_decode(feat_localizations):

    boxes = np.zeros([7,7,2,4])

    cx = feat_localizations[:,:,:,0]
    cy = feat_localizations[:,:,:,1]
    ch = feat_localizations[:,:,:,2]
    cw = feat_localizations[:,:,:,3]

    ymin = cy - ch/2.
    xmin = cx - cw/2.
    ymax = ymin + ch
    xmax = xmin + cw

    boxes[:,:,:,0] = ymin
    boxes[:,:,:,1] = xmin
    boxes[:,:,:,2] = ymax
    boxes[:,:,:,3] = xmax

    return boxes

def yolo_bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            select_threshold=0.5,
                            img_shape=(448, 448),
                            num_classes=21):

    localizations_layer = yolo_boxes_decode(localizations_layer)

    sub_predictions = predictions_layer[:, :, 1:]
    idxes = np.where(sub_predictions > select_threshold)
    classes = idxes[-1]+1
    scores = sub_predictions[idxes]
    bboxes = localizations_layer[idxes[0:2]]

    return classes, scores, bboxes


def yolo_bboxes_select(predictions_net,
                      localizations_net,
                      select_threshold=0.5,
                      img_shape=(448, 448),
                      num_classes=21):
    l_classes = []
    l_scores = []
    l_bboxes = []

    
    classes, scores, bboxes = yolo_bboxes_select_layer(
            predictions_net, localizations_net,
            select_threshold, img_shape, num_classes)
    l_classes.append(classes)
    l_scores.append(scores)
    l_bboxes.append(bboxes)

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    
    classes = np.tile(classes, (2,1))
    scores = np.tile(scores, (2,1))
    classes = np.transpose(classes)
    scores = np.transpose(scores)
    
    bboxes = np.concatenate(l_bboxes, 0)

    return classes, scores, bboxes


def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    # print classes.shape
    # print scores.shape
    # print bboxes.shape
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1) #i
    bboxes2 = np.transpose(bboxes2) #i+1
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]