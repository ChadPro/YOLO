# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from nets import yolo_net   # Create Yolo Net
from datasets import pascalvoc_2012     # Read Data
from preprocessing import yolo_preprocessing    # Data Aug


####################
#   Learn param    #
####################
tf.app.flags.DEFINE_float('learning_rate_base', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.99, 'Decay learning rate.')
tf.app.flags.DEFINE_integer('learning_decay_step', 500, 'Learning rate decay step.')
tf.app.flags.DEFINE_string('train_data_path', '', 'Dataset for train.')
tf.app.flags.DEFINE_bool('fine_tune', False, 'If fine tune.')
tf.app.flags.DEFINE_string('model_store_path', './yolo_model/yolo_model.ckpt', '')
FLAGS = tf.app.flags.FLAGS

'''
    1. Create yolo object
'''
yolo_obj = yolo_net.YOLO()

'''
    2. Get Dataset and Preprocess + Shuffle Batch
'''
# read tfrecord data
image, shape, box, label = pascalvoc_2012.inputs(FLAGS.train_data_path, "", "Train", 1, None)
# preprocess data aug
bimage, blabel, bbox = yolo_preprocessing.preprocess_for_train(image, label, box, (448,448), image_whitened=False, color_distort=True)
# encode
blabels, bboxes, bscores = yolo_obj.encode_boxes(bbox, blabel)
# Shuffle Batch
imagess, boxess, labelss, scoress = tf.train.shuffle_batch([bimage,bboxes,blabels,bscores], batch_size=8, num_threads=2, capacity=1000, min_after_dequeue=200)

'''
    3. Inference
'''
y, weights = yolo_obj.yolo_net(imagess, 21)
yolo_obj.net_loss(y, labelss, boxess, scoress, 8)

# 4. Loss
losses = tf.get_collection(tf.GraphKeys.LOSSES)
total_loss = tf.add_n(losses)
default_graph = tf.get_default_graph()
cls_loss = default_graph.get_tensor_by_name("net_loss/class_loss/mul_1:0")
obj_loss = default_graph.get_tensor_by_name("net_loss/object_loss/mul_1:0")
noobj_loss = default_graph.get_tensor_by_name("net_loss/noobject_loss/mul_1:0")
cor_loss = default_graph.get_tensor_by_name("net_loss/coord_loss/mul_3:0")
tf.summary.scalar('class_loss', cls_loss)
tf.summary.scalar('object_loss', obj_loss)
tf.summary.scalar('noobject_loss', noobj_loss)
tf.summary.scalar('coord_loss', cor_loss)
'''
    5. Train Step
    train_step 梯度下降(学习率，损失函数，全局步数) + BN Layer Params update op
'''
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base, global_step, FLAGS.learning_decay_step, FLAGS.learning_rate_decay)     
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
'''
    6. Saver
'''
merged = tf.summary.merge_all()
logwriter = tf.summary.FileWriter("./log_dir/yolo/", tf.get_default_graph())
basenet_saver = tf.train.Saver(weights)
saver = tf.train.Saver()

'''
    7. Loop
'''
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if FLAGS.fine_tune:
        print "######### restore yolo model ##################"
        saver.restore(sess, FLAGS.model_store_path)
    else:
        print "######### create yolo model ##################"
        basenet_saver.restore(sess, "./base_model/basemodel.ckpt")

    for i in range(1000000):
        summary_str, _, loss_value, step, loss_list, class_loss, object_loss, noobject_loss, coord_loss = sess.run([merged, train_step, total_loss, global_step, losses, cls_loss, obj_loss, noobj_loss, cor_loss])

        if i%30 == 0:
            logwriter.add_summary(summary_str, i)
            # Loss info
            print "cls-loss={0} , obj-loss={1}, noobj-loss={2}, coord-loss={3}".format(str(class_loss), str(object_loss), str(noobject_loss), str(coord_loss))

        if i%500 == 0:
            saver.save(sess, FLAGS.model_store_path)

    logwriter.close()
    coord.request_stop()
    coord.join(threads)