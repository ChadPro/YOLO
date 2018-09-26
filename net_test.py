# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from nets import base_net

yolo_base = base_net.YOLO_Base()

img = tf.ones([32,448,448,3])

net = yolo_base.base_net(img, 1000)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    result = sess.run(net)
    print result.shape