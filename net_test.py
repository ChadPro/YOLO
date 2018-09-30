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

# 1. Create yolo object
yolo_base = base_net.YOLO_Base()
yolo_obj = yolo_net.YOLO()

# 2. Get Dataset and Preprocess + Shuffle Batch
image, shape, box, label = pascalvoc_2012.inputs("../dataset/VOCdevkit/pascal_voc21/", "", "Train", 16, None)
bimage, blabel, bbox = yolo_preprocessing.preprocess_for_train(image, label, box, (448,448))
blabels, bboxes, bscores = yolo_obj.encode_boxes(bbox, blabel)
imagess, boxess, labelss, scoress = tf.train.shuffle_batch([bimage,bboxes,blabels,bscores], batch_size=3, num_threads=64, capacity=5000, min_after_dequeue=3000)

# 3. Inference
y = yolo_obj.yolo_net(imagess, 21)

# loss = yolo_obj.net_loss(y, None, 16)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    out1 = sess.run([y])
    print out1

    coord.request_stop()
    coord.join(threads)