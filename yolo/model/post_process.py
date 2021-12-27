#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import sys

import numpy as np
import tensorflow as tf
from yolo.dataset.image_utils import xywh2xyxy, box_iou

sys.path.append('..')


# import numpy as np


def batch_non_max_suppression(prediction, conf_threshold=0.5, iou_threshold=0.25, classes=None, agnostic=False,
                              labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    prediction: batch_size * 3grid * (num_classes + 5)
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    redundant = True  # require redundant detections

    prediction = [tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1])) for x in prediction]
    prediction = tf.concat(prediction, axis=1)  # batch_size * -1 * (num_class + 5)

    # num_classes = tf.shape(prediction)[-1] - 5
    # candidates = prediction[..., 4] > conf_threshold
    output = [tf.zeros((0, 6))] * prediction.shape[0]

    for i in range(prediction.shape[0]):  # iter for image
        pred = prediction[i]
        pred = pred[pred[:, 4] >= conf_threshold]  # filter by yolo confidence
        if not pred.shape[0]:
            continue

        box = xywh2xyxy(pred[:, :4])
        score = pred[:, 4]
        classes = tf.argmax(pred[:, 5:], axis=1)
        pred_nms = []
        for clss in tf.unique(classes)[0]:
            mask = tf.math.equal(classes, clss)  # compare
            box_of_clss = tf.boolean_mask(box, mask)  # n_conf * 4
            classes_of_clss = tf.boolean_mask(classes, mask)  # n_conf
            score_of_clss = tf.boolean_mask(score, mask)  # n_conf

            select_indices = tf.image.non_max_suppression(box_of_clss, score_of_clss, max_output_size=50,
                                                                   iou_threshold=iou_threshold)  # for one class

            box_of_clss = tf.gather(box_of_clss, select_indices)
            score_of_clss = tf.gather(tf.expand_dims(score_of_clss, -1), select_indices)
            classes_of_clss = tf.cast(tf.gather(tf.expand_dims(classes_of_clss, -1), select_indices), tf.float32)
            pred_of_clss = tf.concat([box_of_clss, score_of_clss, classes_of_clss], axis=-1)

            pred_nms.append(pred_of_clss)

        output[i] = tf.concat(pred_nms, axis=0)
    return output


def weighted_boxes_fusion():
    return

# def batch_non_max_suppression(prediction,
# conf_threshold=0.1, iou_threshold=0.6, merge=False, classes=None, agnostic=False):
#     """
#     Performs Non-Maximum Suppression (NMS) on inference results
#
#     Returns:
#          detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
#     """
#     prediction = [tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1])) for x in prediction]
#     prediction = tf.concat(prediction, axis=1)  # batch_size * -1 * (num_class + 5)
#
#     nc = prediction[0].shape[1] - 5  # number of classes
#     xc = prediction[..., 4] > conf_threshold  # candidates
#
#     # Settings
#     min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
#     max_det = 300  # maximum number of detections per image
#     time_limit = 10.0  # seconds to quit after
#     redundant = True  # require redundant detections
#     multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
#
#     output = [None] * prediction.shape[0]
#     for xi, x in enumerate(prediction):  # image index, image inference
#         # Apply constraints
#         # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#         x = x[xc[xi]]  # confidence
#
#         # If none remain process next image
#         if not x.shape[0]:
#             continue
#
#         # Compute conf
#         x1 = x.numpy()
#         x1[:, 5:] *= x1[:, 4:5]  # conf = obj_conf * cls_conf
#         x = tf.convert_to_tensor(x1)
#         # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#         box = xywh2xyxy(x[:, :4])
#
#         # Detections matrix nx6 (xyxy, conf, cls)
#         if multi_label:
#             i, j = (x[:, 5:] > conf_threshold).nonzero(as_tuple=False).T
#             x = tf.concat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
#         else:  # best class only
#             conf, j = x[:, 5:].max(1, keepdim=True)
#             x = tf.concat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
#
#         # Filter by class
#         if classes:
#             x = x[(x[:, 5:6] == tf.constant(classes)).any(1)]
#
#         # Apply finite constraint
#         # if not torch.isfinite(x).all():
#         #     x = x[torch.isfinite(x).all(1)]
#
#         # If none remain process next image
#         n = x.shape[0]  # number of boxes
#         if not n:
#             continue
#
#         # Sort by confidence
#         # x = x[x[:, 4].argsort(descending=True)]
#
#         # Batched NMS
#         c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#         boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#         i = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=iou_threshold)
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]
#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#                 iou = box_iou(boxes[i], boxes) > iou_threshold  # iou matrix
#                 weights = iou * scores[None]  # box weights
#                 x[i, :4] = tf.multiply(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#                 if redundant:
#                     i = i[iou.sum(1) > 1]  # require redundancy
#             except:
#                 # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
#                 print(x, i, x.shape, i.shape)
#                 pass
#
#         output[xi] = x[i]
#
#     return output


