# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

STDDEV = 0.01
VGG_MEAN = [122.173, 116.150, 103.504]  # bgr
DEFAULT_OUTPUT_NODE = 1000
BN_DECAY = 0.9
ACTIVATION = tf.nn.relu

class YOLO_Base(object):

    def __init__(self):
        pass
    
    def base_net(self, inputs, num_classes, is_training=True):
        return yolo_base_net(inputs, num_classes)

    def net_loss(self, y, label):
        pass

def yolo_base_net(inputs, num_classes, is_training=True):
    conv_num = 1

    net, w1, b1 = conv_block(inputs, [7,7,3,64], [1,2,2,1], "conv"+str(conv_num), is_training=is_training)
    net = max_pool_block(net, [1,2,2,1], [1,2,2,1], "max_pool1", is_training=is_training)
    conv_num = conv_num + 1

    net, w2, b2 = conv_block(net, [3,3,64,192], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    net = max_pool_block(net, [1,2,2,1], [1,2,2,1], "max_pool2", is_training=is_training)
    conv_num = conv_num + 1

    net, w3, b3 = conv_block(net, [1,1,192,128], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    conv_num = conv_num + 1
    net, w4, b4 = conv_block(net, [3,3,128,256], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    conv_num = conv_num + 1
    net, w5, b5 = conv_block(net, [1,1,256,256], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    conv_num = conv_num + 1
    net, w6, b6 = conv_block(net, [3,3,256,512], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    conv_num = conv_num + 1
    net = max_pool_block(net, [1,2,2,1], [1,2,2,1], "max_pool3", is_training=is_training)

    for i in range(4):
        net, w_1, b_1 = conv_block(net, [1,1,512,256], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
        conv_num = conv_num + 1
        net, w_2, b_2 = conv_block(net, [3,3,256,512], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
        conv_num = conv_num + 1
    net, w15, b15 = conv_block(net, [1,1,512,512], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    conv_num = conv_num + 1
    net, w16, b16 = conv_block(net, [3,3,512,1024], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    conv_num = conv_num + 1
    net = max_pool_block(net, [1,2,2,1], [1,2,2,1], "max_pool4", is_training=is_training)

    for i in range(2):
        net, w_1, b_1 = conv_block(net, [1,1,1024,512], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
        conv_num = conv_num + 1
        net, w_2, b_2 = conv_block(net, [3,3,512,1024], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
        conv_num = conv_num + 1
    net, w21, b21 = conv_block(net, [3,3,1024,1024], [1,1,1,1], "conv"+str(conv_num), is_training=is_training)
    conv_num = conv_num + 1
    net, w22, b22 = conv_block(net, [3,3,1024,1024], [1,2,2,1], "conv"+str(conv_num), is_training=is_training)
    
    net = avg_pool_block(net, [1,4,4,1], [1,4,4,1], "avg_pool", is_training=is_training)

    pool_shape = net.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(net, [pool_shape[0], nodes])


    with tf.variable_scope('layer-fc1'):
        fc1_weights = tf.get_variable("fc_w1", [nodes, num_classes], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        fc1_biase = tf.get_variable('fc_b1', [num_classes], initializer=tf.constant_initializer(0.0))
        fc1 = tf.matmul(reshaped, fc1_weights) + fc1_biase

    return fc1


def avg_pool_block(inputs, p_size, strides, scope_name, is_training=True):
    with tf.name_scope(scope_name):
        output = tf.nn.avg_pool(inputs, p_size, strides, padding='SAME', name="avg_pool")
    return output

def max_pool_block(inputs, p_size, strides, scope_name, is_training=True):
    with tf.name_scope(scope_name):
        output = tf.nn.max_pool(inputs, p_size, strides, padding='SAME', name="max_pool")
    return output

def conv_block(inputs, w_size, strides, scope_name, is_training=True):
    with tf.variable_scope(scope_name):
        weights = tf.get_variable("dw", w_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        biases = tf.get_variable("biases", w_size[-1], initializer=tf.constant_initializer(0.0))
        layer = tf.nn.conv2d(inputs, weights, strides=strides, padding='SAME')
        bn = tf.contrib.layers.batch_norm(tf.nn.bias_add(layer, biases), decay=BN_DECAY, center=True, scale=True, is_training=is_training, scope='dw_bn')
        output = ACTIVATION(bn)
    return output, weights, biases

