# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import cv2
from nets import base_net
from nets import yolo_net
from preprocessing import yolo_preprocessing
from datasets import pascalvoc_2012
from py_extend import np_methods

tf.app.flags.DEFINE_string('image_path', '', 'Image path.')
FLAGS = tf.app.flags.FLAGS

# 1. Create yolo object
yolo_base = base_net.YOLO_Base()
yolo_obj = yolo_net.YOLO()

# 2. Inference
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
img_pre = yolo_preprocessing.preprocess_for_detect(img_input, None, None, [448,448])
image_4d = tf.expand_dims(img_pre, 0)
y, weights = yolo_obj.yolo_net(image_4d, 21)

# 3. Read Image
img = cv2.imread(FLAGS.image_path)
ssd_saver = tf.train.Saver()

def cal_iou(box1, box2):
    xmin = np.maximum(box1[1], box2[1])
    ymin = np.maximum(box1[0], box2[0])
    xmax = np.minimum(box1[3], box2[3])
    ymax = np.minimum(box1[2], box2[2])

    s1 = (box1[3]-box1[1])*(box1[2]-box1[0])
    s2 = (box2[3]-box2[1])*(box2[2]-box2[0])

    w = np.maximum(xmax-xmin, 0.)
    h = np.maximum(ymax-ymin, 0.)
    inter = w * h
    union = np.maximum(s1+s2-inter,0.1)
    
    return inter / union

# 4. Tf Session()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    ssd_saver.restore(sess, './yolo_model/yolo_model.ckpt')

    r, r_image = sess.run([y, img_pre], feed_dict={img_input : img})

    r_reshape = np.reshape(r, [1,7,7,31])
    predict_classes = r_reshape[:,:,:,:21]  # (1,7,7,21)
    predict_boxes = r_reshape[:,:,:,21:29]  # (1,7,7,8)
    predict_boxes = np.reshape(predict_boxes, [1,7,7,2,4])
    predict_scales = r_reshape[:,:,:,29:]   # (1,7,7,2)

    # box decode
    anchor_offset = np.transpose(np.reshape(np.array([np.arange(7)]*7*2), (2,7,7)), (1,2,0))
    anchor_offset = np.reshape(np.array(anchor_offset, dtype=float), [1,7,7,2])
    anchor_offset_tran = np.transpose(anchor_offset, (0,2,1,3))
    predict_boxes = np.array([predict_boxes[..., 0],
                                predict_boxes[...,1],
                                np.square(predict_boxes[...,2]),
                                np.square(predict_boxes[...,3])])
    # predict_boxes = np.array([(predict_boxes[..., 0]+anchor_offset)/7,
    #                             (predict_boxes[...,1]+anchor_offset_tran)/7,
    #                             np.square(predict_boxes[...,2]),
    #                             np.square(predict_boxes[...,3])])                                
    # predict_boxes = np.array([predict_boxes[1]-predict_boxes[3]/2.,
    #                             predict_boxes[0]-predict_boxes[2]/2.,
    #                             predict_boxes[1]+predict_boxes[3],
    #                             predict_boxes[0]+predict_boxes[2]])
    predict_boxes = np.array([predict_boxes[1]-predict_boxes[3]/2.,
                                predict_boxes[0]-predict_boxes[2]/2.,
                                predict_boxes[1]+predict_boxes[3]/2.,
                                predict_boxes[0]+predict_boxes[2]/2.])

    # predict_cls = np.argmax(predict_classes, axis=3)
    # print predict_cls

    predict_classes = np.reshape(predict_classes, (1,7,7,1,21))
    predict_classes = np.tile(predict_classes, [1,1,1,2,1])
    predict_boxes = np.reshape(predict_boxes, [1,7,7,2,1,4])
    predict_boxes = np.tile(predict_boxes, [1,1,1,1,21,1])
    predict_scales = np.reshape(predict_scales, (1,7,7,2,1))
    predict_scales = np.tile(predict_scales, [1,1,1,1,21])

    predict_boxes = np.reshape(predict_boxes, (-1,4))
    predict_classes = np.reshape(predict_classes, (-1))
    predict_scales = np.reshape(predict_scales, (-1))

    predict_objs = np.arange(21)
    predict_objs = np.tile(predict_objs, 7*7*2)
    predict_classes = predict_classes * predict_scales

    l_objs = predict_objs
    l_scores = predict_classes
    l_coors = predict_boxes

    iou_thread = 0.6
    nms_objs = []
    nms_scores = []
    nms_coors = []

    score_index = np.where(l_scores > 0.3)
    l_objs = l_objs[score_index]
    l_scores = l_scores[score_index]
    l_coors = l_coors[score_index]

    while len(l_objs) > 0:
        max_index = np.argmax(l_scores)
        max_obj = l_objs[max_index]
        max_score = l_scores[max_index]
        max_coor = l_coors[max_index]
        nms_objs.append(max_obj)
        nms_scores.append(max_score)
        nms_coors.append(max_coor)

        l_objs = np.delete(l_objs, [max_index])
        l_scores = np.delete(l_scores, [max_index])
        l_coors = np.delete(l_coors, [max_index], axis=0)

        delete_l = []
        for j, select_coor in enumerate(l_coors):
            iou = cal_iou(max_coor, select_coor)
            if iou > iou_thread:
                delete_l.append(j)
        l_objs = np.delete(l_objs, delete_l)
        l_scores = np.delete(l_scores, delete_l)
        l_coors = np.delete(l_coors, delete_l, axis=0)

    print nms_objs
    print nms_scores
    print nms_coors

    for jj, hehe in enumerate(nms_coors):
        if nms_objs[jj] > 0:
            cv2.rectangle(r_image, (int(hehe[1]*448), int(hehe[0]*448)), (int(hehe[3]*448), int(hehe[2]*448)), (0,0,255), 2)
            cv2.putText(r_image, str(nms_objs[jj]), (int(hehe[1]*448),int(hehe[0]*448)-10), 2, 1.5, (0, 0, 255))
    cv2.imwrite("./hehe.jpg", r_image)

    # # just for test not nms
    # p_classes = np.reshape(predict_classes, (7,7,1,21))
    # C = np.reshape(predict_scales, (7,7,2,1))
    # P = C * p_classes

    # index = np.argmax(P)
    # index = np.unravel_index(index, P.shape)
    # class_num = index[3]
    # coordinate = np.reshape(predict_boxes, (7,7,2,4))
    # max_coordinate = coordinate[index[0], index[1], index[2], :]
    # xcenter = max_coordinate[0]
    # ycenter = max_coordinate[1]
    # w = max_coordinate[2]
    # h = max_coordinate[3]

    # xcenter = (index[1]+xcenter)*(448/7.0)
    # ycenter = (index[0]+ycenter)*(448/7.0)

    # w = w * 448
    # h = h * 448

    # xmin = xcenter - w/2.0
    # ymin = ycenter - h/2.0

    # xmax = xmin + w
    # ymax = ymin + h

    # # hehe = predict_boxes[0]
    # # d_class, d_score, d_box = np_methods.yolo_bboxes_select(predict_classes[0],hehe,select_threshold=0.3)
    # # r_boxes = np_methods.bboxes_clip([0.,0.,1.,1.], d_box)

    # # d_class = np.reshape(d_class, (-1,))
    # # d_score = np.reshape(d_score, (-1))
    # # r_boxes = np.reshape(r_boxes, (-1,4))

    # # d_class, d_score, r_boxes = np_methods.bboxes_sort(d_class, d_score, r_boxes)
    # # d_class, d_score, r_boxes = np_methods.bboxes_nms(d_class, d_score, r_boxes)

    # # max_index = np.argmax(d_score)
    # # d_ymin = int(r_boxes[max_index][0] * 448)
    # # d_xmin = int(r_boxes[max_index][1] * 448)
    # # d_ymax = int(r_boxes[max_index][2] * 448)
    # # d_xmax = int(r_boxes[max_index][3] * 448)

    # print("###### detect #########")
    # print("detect cls = " + str(class_num))

    # cv2.rectangle(r_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,0,0), 2)
    # cv2.imwrite("./hehe.jpg", r_image)


