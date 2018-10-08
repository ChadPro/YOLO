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

# 0. Data for Test
# test_image = tf.ones([16,448,448,3])
# test_boxes = tf.zeros([16,7,7,1,4])
# test_labels = tf.zeros([16,7,7,1],dtype=tf.int64)
# test_scores = tf.zeros([16,7,7,1])

# 1. Create yolo object
yolo_base = base_net.YOLO_Base()
yolo_obj = yolo_net.YOLO()

# # 2. Get Dataset and Preprocess + Shuffle Batch
image, shape, box, label = pascalvoc_2012.inputs("../dataset/VOCdevkit/pascal_voc21/", "", "Train", 16, None)
bimage, blabel, bbox = yolo_preprocessing.preprocess_for_train(image, label, box, (448,448))
blabels, bboxes, bscores = yolo_obj.encode_boxes(bbox, blabel)
imagess, boxess, labelss, scoress = tf.train.shuffle_batch([bimage,bboxes,blabels,bscores], batch_size=16, num_threads=64, capacity=5000, min_after_dequeue=3000)

# 3. Inference
y = yolo_obj.yolo_net(imagess, 21)
yolo_obj.net_loss(y, labelss, boxess, scoress, 16)

losses = tf.get_collection(tf.GraphKeys.LOSSES)
total_loss = tf.add_n(losses)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.0001, global_step, 1000, 0.99)  
    #train_step 梯度下降(学习率，损失函数，全局步数) + BN Layer Params update op
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

merged = tf.summary.merge_all()
logwriter = tf.summary.FileWriter("./log_dir/yolo/", tf.get_default_graph())

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(10000):
        summary_str, _, loss_value, step, loss_list = sess.run([merged, train_step, total_loss, global_step, losses])

        if i%30 == 0:
            logwriter.add_summary(summary_str, i)
            print loss_list
    logwriter.close()
    coord.request_stop()
    coord.join(threads)