# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import cv2
from nets import base_net
from nets import yolo_net
from preprocessing import yolo_preprocessing
from datasets import pascalvoc_2012
from py_extend import np_methods

tf.app.flags.DEFINE_string('image_path', '', 'Image path.')
FLAGS = tf.app.flags.FLAGS

# 1. Create yolo object
yolo_base = base_net.YOLO_Base()
yolo_obj = yolo_net.YOLO()

# 2. Inference
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
img_pre = yolo_preprocessing.preprocess_for_detect(img_input, None, None, [448,448])
image_4d = tf.expand_dims(img_pre, 0)
y = yolo_obj.yolo_net(image_4d, 21)

# 3. Read Image
img = cv2.imread("./demo/person1.jpg")
ssd_saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ssd_saver.restore(sess, './yolo_model/yolo_model.ckpt')

    r = sess.run(y, feed_dict={img_input : img})
    r_reshape, r_image = np.reshape([r, img_pre], [1,7,7,31])
    predict_classes = r_reshape[:,:,:,:21]
    predict_boxes = r_reshape[:,:,:,21:29]
    predict_boxes = np.reshape(predict_boxes, [1,7,7,2,4])
    predict_scales = r_reshape[:,:,:,29:]

    hehe = predict_boxes[0]
    d_class, d_score, d_box = np_methods.yolo_bboxes_select(predict_classes[0],hehe,select_threshold=0.3)
    r_boxes = np_methods.bboxes_clip([0.,0.,1.,1.], d_box)

    d_class = np.reshape(d_class, (-1,))
    d_score = np.reshape(d_score, (-1))
    r_boxes = np.reshape(r_boxes, (-1,4))

    d_class, d_score, r_boxes = np_methods.bboxes_sort(d_class, d_score, r_boxes)
    d_class, d_score, r_boxes = np_methods.bboxes_nms(d_class, d_score, r_boxes)

    max_index = np.argmax(d_score)
    d_ymin = int(r_boxes[max_index][0] * 448)
    d_xmin = int(r_boxes[max_index][1] * 448)
    d_ymax = int(r_boxes[max_index][2] * 448)
    d_xmax = int(r_boxes[max_index][3] * 448)

    cv2.rectangle(r_image, (d_xmin, d_ymin), (d_xmax, d_ymax), (255,0,0), 2)
    cv2.imwrite("./hehe.jpg", r_image)


