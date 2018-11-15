# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import cv2
import argparse

from PIL import Image, ImageDraw

from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f.readlines()):
            names[id] = str(name).replace('\n', '')
    return names


def draw_boxes(boxes, img, cls_names, detection_size, colors):
    for cls, bboxs in boxes.items():
        color = colors[cls]
        color = (int(color[0]), int(color[1]), int(color[2]))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array([img.shape[1], img.shape[0]]))
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img, pt1, pt2, color, 2)
            text = '{} {:.2f}%'.format(str(cls_names[cls]), score * 100)
            cv2.putText(img, text, pt1, 2, 1.2, color, 2)
    return img


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('camid', type=int, help='source webcam id')
    args = parser.parse_args()
    classes = load_coco_names(FLAGS.class_names)
    np.random.seed(2018)
    colors = [np.random.randint(0, 255, 3) for _ in range(len(classes))]

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes), data_format='NCHW')
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    boxes = detections_boxes(detections)
    vc = cv2.VideoCapture()
    vc.open(args.camid)
    with tf.Session() as sess:
        sess.run(load_ops)
        while True:
            _, img = vc.read()
            img_resized = cv2.resize(img, dsize=(FLAGS.size, FLAGS.size))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
            filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
                                                 iou_threshold=FLAGS.iou_threshold)
            img = cv2.resize(img, (1920, 1080))
            img = draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), colors=colors)
            # img_resized = draw_boxes(filtered_boxes, img_resized, classes, (FLAGS.size, FLAGS.size))
            cv2.imshow('detections', img)
            # cv2.imshow('img_resized', img_resized)
            key = cv2.waitKey(1)
            if key == ord('q') or key & 0xFFFF == 27:
                break


if __name__ == '__main__':
    tf.app.run()
