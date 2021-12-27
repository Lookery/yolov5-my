#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com

import sys
import os
import cv2
# import time
import argparse
import numpy as np
import tensorflow as tf
from dataset.image_utils import resize_image, resize_back
from model.post_process import batch_non_max_suppression
from tools.vis_data import draw_box

filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])


def image_demo(img, model, i=0, img_size=640, class_names=None, conf_threshold=0.6, iou_threshold=0.25):
    original_shape = img.shape

    img_input = resize_image(img, target_sizes=img_size)
    img_input = img_input[np.newaxis, ...].astype(np.float32)
    img_input = img_input / 255.

    pred_bbox = model(img_input)

    bboxes = batch_non_max_suppression(pred_bbox, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    bboxes = bboxes[0].numpy()  # batch is 1 for detect

    bboxes = resize_back(bboxes, target_sizes=img_size, original_shape=original_shape)  # adjust box to original size

    if bboxes.any():
        print(bboxes)
        image = draw_box(img, np.array(bboxes), class_names)
        # cv2.imshow('imshow', img)
        # cv2.waitKey(0)
        cv2.imwrite(base_dir + '/data/results/demo%d.jpg' % i, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        print('No box detected')


def video_demo(video_path, model,
               img_size=640, class_names=None,
               conf_threshold=0.6, iou_threshold=0.25):
    # video_path      = 0

    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image_data = Image.fromarray(frame)
        else:
            raise ValueError("No image!")
        # frame_size = frame.shape[:2]
        # image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        # image_data = image_data[np.newaxis, ...]
        # prev_time = time.time()
        original_shape = frame.shape
        img_input = resize_image(np.copy(frame), target_sizes=img_size)
        img_input = img_input[np.newaxis, ...].astype(np.float32)
        img_input = img_input / 255.

        pred_bbox = model(img_input)

        bboxes = batch_non_max_suppression(pred_bbox, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        bboxes = bboxes[0].numpy()  # batch is 1 for detect
        bboxes = resize_back(bboxes, target_sizes=img_size,
                             original_shape=original_shape)  # adjust box to original size
        if bboxes.any():
            image = draw_box(frame, np.array(bboxes), class_names)

        else:
            image = frame


        # curr_time = time.time()
        # exec_time = curr_time - prev_time
        # result = np.asarray(image)
        # info = "time: %.2f ms" % (1000 * exec_time)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def test_image_demo(img_dir, model_dir, img_size=640, class_name_dir=None, conf_threshold=0.4, iou_threshold=0.3):
    model = tf.keras.models.load_model(model_dir, compile=False)
    if class_name_dir:
        class_names = {idx: name for idx, name in enumerate(open(class_name_dir).read().splitlines())}
    else:
        class_names = None
    for i in range(1, 100):
        try:
            img = cv2.imread(img_dir + str(i).rjust(6, '0') + '.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            break

        image_demo(img, model, i=i, img_size=img_size, class_names=class_names,
                   conf_threshold=conf_threshold, iou_threshold=iou_threshold)


def test_video_demo(video_dir, model_dir, img_size=640, class_name_dir=None, conf_threshold=0.4, iou_threshold=0.3):
    model = tf.keras.models.load_model(model_dir, compile=False)
    if class_name_dir:
        class_names = {idx: name for idx, name in enumerate(open(class_name_dir).read().splitlines())}
    else:
        class_names = None

    video_demo(video_dir, model, img_size=img_size, class_names=class_names,
               conf_threshold=conf_threshold, iou_threshold=iou_threshold)


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',
                        type=str, help='detect image dir',
                        default=base_dir + '/data/voc/test/VOCdevkit/VOC2007/JPEGImages/')
    parser.add_argument('--video_dir',
                        type=str, help='detect video dir',
                        default=base_dir + '/data/sample/467488924-1-208.mp4')
    parser.add_argument('--class_name_dir',
                        type=str, default=base_dir + '/data/voc/voc.names',
                        help='classes name dir')
    parser.add_argument('--model_dir',
                        type=str, default=base_dir + '/weights/yolov5',
                        help='saved pb model dir')
    parser.add_argument('--img_size', type=int, default=640, help='image target size')
    parser.add_argument('--conf_threshold', type=float, default=0.4, help='filter confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='nms iou threshold')
    opt = parser.parse_args()
    # test_image_demo(opt.img_dir, opt.model_dir, opt.img_size,
    # opt.class_name_dir, opt.conf_threshold, opt.iou_threshold)
    test_video_demo(opt.video_dir, opt.model_dir,
                    img_size=opt.img_size,
                    class_name_dir=opt.class_name_dir,
                    conf_threshold=opt.conf_threshold,
                    iou_threshold=opt.iou_threshold)
