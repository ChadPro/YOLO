# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from nets import yolo_net   # Create Yolo Net
from datasets import pascalvoc_2012     # Read Data
from preprocessing import yolo_preprocessing    # Data Aug
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

'''
    1. Create yolo object
'''
yolo_obj = yolo_net.YOLO()

'''
    2. Get Dataset and Preprocess + Shuffle Batch
'''
# read tfrecord data
image, shape, box, label = pascalvoc_2012.inputs("../voc_data/voc21/", "", "Train", 1, None)
# preprocess data aug
bimage, blabel, bbox = yolo_preprocessing.preprocess_for_train(image, label, box, (448,448),image_whitened=False, color_distort=False)
# encode
blabels, bboxes, bscores = yolo_obj.encode_boxes(bbox, blabel)
# Shuffle Batch
imagess, boxess, labelss, scoress = tf.train.shuffle_batch([bimage,bboxes,blabels,bscores], batch_size=8, num_threads=2, capacity=1000, min_after_dequeue=200)


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #
    # b_boxes, b_labels, b_scores = sess.run([boxess, labelss, scoress])
    # print b_scores
    # print b_boxes.shape
    # print b_labels.shape
    # print b_scores.shape

    # draw
    r_image, o_box, r_scores, r_box, r_label = sess.run([bimage, bbox, bscores, bboxes, blabels])
    print r_scores
    print r_scores.shape
    fig, ax = plt.subplots(1)
    for i in range(8):
        plt.plot([0,7], [i,i], color='black')
        plt.plot([i,i], [0,7], color='black')
    for bbox in o_box:
        cv2.rectangle(r_image, (int(bbox[1]*448),int(bbox[0]*448)), (int(bbox[3]*448),int(bbox[2]*448)),(0,0,255), 2)
    cv2.imwrite("process.jpg", r_image)
    
    for i_x in range(7):
        for i_y in range(7):
            xy = r_box[i_x][i_y][0]
            plt.text(i_x+0.5, 7-i_y-0.5, str(r_label[i_y][i_x][0]))
            if (xy[1] == 0.)&(xy[0] == 0.)&(xy[2] == 1.)&(xy[3] == 1.):
                pass
            else:
                rect = patches.Rectangle((xy[0]*7,7-xy[1]*7), xy[2]*7, -xy[3]*7)
                # ax.add_patch(rect)
    plt.show()

    coord.request_stop()
    coord.join(threads)
