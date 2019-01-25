# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from nets import base_net_bn
from nets import yolo_net
from preprocessing import yolo_preprocessing
from datasets import pascalvoc_2012 

# 0. Data for Test
IMG_SIZE = 448
test_image = tf.ones([16,IMG_SIZE,IMG_SIZE,3])
# test_boxes = tf.zeros([16,7,7,1,4])
# test_labels = tf.zeros([16,7,7,1],dtype=tf.int64)
# test_scores = tf.zeros([16,7,7,1])

# 1. Create yolo object
yolo_obj = yolo_net.YOLO()
y, weights = yolo_obj.yolo_net(test_image, 21)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    r = sess.run(y)
    print r.shape

    coord.request_stop()
    coord.join(threads)
