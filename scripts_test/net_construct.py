# -*- coding=utf-8 -*-
import sys
import tensorflow as tf
sys.path.append("../")
from nets import yolo_net

yolo_obj = yolo_net.YOLO()

imgs = tf.ones([32, 224, 224, 3])
net_out, _ = yolo_obj.yolo_net(imgs, 21)

print "####### net output shape #########"
print net_out.shape

print "####### after reshape output ########"
y = tf.reshape(net_out, [32, 7, 7, 31])
print y.shape

