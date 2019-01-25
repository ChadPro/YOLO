# -*- coding=utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
sys.path.append("../")
from preprocessing import yolo_preprocessing
from datasets import pascalvoc_2012

DEFAULT_DATASETS = "../../voc_data/voc21/"
image, shape, box, label = pascalvoc_2012.inputs(DEFAULT_DATASETS, "", "Train", 1, None)
# preprocess
bimage, blabel, bbox = yolo_preprocessing.preprocess_for_train(image, label, box, (448,448), image_whitened=False, color_distort=True)

print "########### dataset input type #############"
print type(bimage)
print type(bbox)
print type(blabel)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    r_bimage, r_bbox, r_blabel = sess.run([bimage, bbox, blabel])

    print "############ preprocessed data #################"
    print "image.shape=",r_bimage.shape,", data.max=",np.max(r_bimage),", data.min=",np.min(r_bimage)
    print "box.shape=",r_bbox.shape,", box=",r_bbox
    print "label.shape",r_blabel

    row, col = 448, 448

    for i, one_box in enumerate(r_bbox):
        cv2.rectangle(r_bimage, (int(one_box[1]*col), int(one_box[0]*row)), (int(one_box[3]*col), int(one_box[2]*row)), 5)
        cv2.putText(r_bimage, str(r_blabel[i]), (int(one_box[1]*col), int(one_box[0]*row-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv2.imwrite("./image_preprocess.jpg", r_bimage) 

    coord.request_stop()
    coord.join(threads)
    


