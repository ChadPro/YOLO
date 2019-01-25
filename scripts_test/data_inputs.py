# -*- coding=utf-8 -*-
import sys
import cv2
import numpy as np
sys.path.append("../")
import tensorflow as tf
from datasets import pascalvoc_2012

DEFAULT_DATASETS = "../../voc_data/voc21/"

image, shape, box, label = pascalvoc_2012.inputs(DEFAULT_DATASETS, "", "Train", 1, None)

print "########### dataset input type #############"
print type(image)
print type(shape)
print type(box)
print type(label)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
  
    r_image, r_shape, r_box, r_label = sess.run([image, shape, box, label])
    
    print "########### dataset input #################"
    print "image.shape=",r_image.shape,", data.max=",np.max(r_image),", data.min=",np.min(r_image)
    print "shape = ", r_shape 
    print "box.shape=",r_box.shape,", box=",r_box
    print "r_label.shape=",r_label.shape,", r_label=",r_label

    row, col = r_shape[0], r_shape[1]
      
    for i, bbox in enumerate(r_box):
        cv2.rectangle(r_image, (int(bbox[1]*col), int(bbox[0]*row)), (int(bbox[3]*col), int(bbox[2]*row)), 5)
        cv2.putText(r_image, str(r_label[i]), (int(bbox[1]*col), int(bbox[0]*row-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv2.imwrite("./image.jpg", r_image) 

    coord.request_stop()
    coord.join(threads)
   
