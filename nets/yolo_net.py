# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from nets import yolo_common

IMAGE_SIZE = 448
NUM_CHANNELS = 3
STDDEV = 0.01
VGG_MEAN = [122.173, 116.150, 103.504]  # bgr
DEFAULT_OUTPUT_NODE = 1000
BN_DECAY = 0.9
ACTIVATION = tf.nn.relu

class YOLO(object):

    def __init__(self):
        pass
    
    def yolo_net(self, inputs, num_classes, is_training=True):
        return get_yolo_net(inputs, num_classes)

    def net_loss(self, predicts, labels, batch_size, scope="net_loss"):
        return get_layer_loss(predicts, batch_size, scope=scope)

    def detect_anchors(self, image_shape, step, anchor_size=7, offset=0.5, dtype=np.float32):
        y, x = np.mgrid[0:anchor_size, 0:anchor_size]
        # y = (y.astype(dtype) + offset) * step / anchor_size
        # x = (x.astype(dtype) + offset) * step / anchor_size
        y = y.astype(dtype) / anchor_size
        x = x.astype(dtype) / anchor_size
        

    
    def encode_boxes(self, boxes, labels, anchor_size=7, dtype=np.float32):
        y, x = np.mgrid[0:anchor_size, 0:anchor_size]
        ymin = y.astype(dtype) / anchor_size
        xmin = x.astype(dtype) / anchor_size
        ymax = (y.astype(dtype) + 1.) / anchor_size
        xmax = (x.astype(dtype) + 1.) / anchor_size

        ymin = tf.expand_dims(ymin, -1)
        xmin = tf.expand_dims(xmin, -1)
        ymax = tf.expand_dims(ymax, -1)
        xmax = tf.expand_dims(xmax, -1)

        feat_labels, feat_localizations, feat_scores = yolo_common.tf_yolo_bboxes_encode_layer(labels, boxes, [ymin, xmin, ymax, xmax], 21)

        return feat_labels, feat_localizations, feat_scores


def get_yolo_net(inputs, num_classes, is_training=True):
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
    
    net = flatten(net)
    fc1 = fc_block(net, 4096, 1, "fc_layer1", activation=ACTIVATION, is_training=is_training)
    output_layer = fc_block(fc1, 7*7*31, 2, "output_layer", is_training=is_training)

    return output_layer



# def get_layer_loss(predicts, labels, boxes, batch_size, scope="net_loss"):

#     with tf.name_scope(scope):
#         predict_classes = tf.reshape(predicts[:, :7*7*21], [batch_size, 7, 7, 21])
#         predict_score = tf.reshape(predicts[:, 7*7*21:7*7*23], [batch_size, 7, 7, 2])
#         predict_boxes = tf.reshape(predicts[:, 7*7*23:], [batch_size, 7, 7, 2, 4])
    

def get_layer_loss(predicts, batch_size, scope="net_loss"):

    with tf.name_scope(scope):
        predict_classes = tf.reshape(predicts[:, :7*7*21], [batch_size, 7, 7, 21])
        predict_score = tf.reshape(predicts[:, 7*7*21:7*7*23], [batch_size, 7, 7, 2])
        predict_boxes = tf.reshape(predicts[:, 7*7*23:], [batch_size, 7, 7, 2, 4])

        reshaped = tf.reshape(predicts, [batch_size,7,7,31])
    return reshaped

def fc_block(inputs, num_out, id, scope_name, activation=None, is_training=True):
    num_in = inputs.get_shape().as_list()[-1]
    with tf.name_scope(scope_name):
        w = tf.get_variable("fc_w"+str(id), [num_in, num_out], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        b = tf.get_variable("fc_b"+str(id), [num_out], initializer=tf.constant_initializer(0.0))
        fc = tf.matmul(inputs,w) + b
        if activation:
            fc = activation(fc)
    return fc


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

def flatten(x):
    tran_x = tf.transpose(x, [0, 3, 1, 2])
    nums = np.product(x.get_shape().as_list()[1:])
    return tf.reshape(tran_x, [-1, nums])