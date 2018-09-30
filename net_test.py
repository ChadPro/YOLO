# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from nets import base_net
from nets import yolo_net
from preprocessing import yolo_preprocessing
from datasets import pascalvoc_2012 

image, shape, box, label = pascalvoc_2012.inputs("./pascal_voc21/", "", "Train", 16, None)
bimage, bbox, blabel = yolo_preprocessing.preprocess_for_train(image, label, box, (448,448))

yolo_base = base_net.YOLO_Base()
yolo_obj = yolo_net.YOLO()

# img = tf.ones([16,448,448,3])
# boxes_input = tf.constant([[0.2, 0.2, 0.4, 0.4], [0.4,0.4,0.6,0.6]])
# labels_input = tf.constant([11, 6], dtype=tf.int64)

blabels, bboxes, bscores = yolo_obj.encode_boxes(bbox, blabel)

imagess, boxess, labelss, scoress = tf.train.shuffle_batch([bimage,bboxes,blabels,bscores], batch_size=3, num_threads=64, capacity=500, min_after_dequeue=3000)

# net = yolo_obj.yolo_net(img, 21)
# output = yolo_obj.net_loss(net, None, 16)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    out1, out2, out3, out4 = sess.run([imagess, boxess, labelss, scoress])
    print out2
    print out3
    print out4