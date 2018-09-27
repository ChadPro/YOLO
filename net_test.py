# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from nets import base_net

yolo_base = base_net.YOLO_Base()

label = tf.constant([[0.5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0., 0., 0.]])
label2 = tf.constant([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

img = tf.ones([1,448,448,3])

net = yolo_base.base_net(img, 17)

ss = tf.nn.softmax(net)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label2,logits=label2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    result, out = sess.run([ss, net])
    print out
    print result