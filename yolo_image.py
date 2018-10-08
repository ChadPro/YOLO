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
    print r.shape

