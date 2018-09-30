# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import tf_extended as tfe
from tensorflow.python.ops import control_flow_ops
from preprocessing import tf_image

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.8,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):

    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=BBOX_CROP_OVERLAP,
                                                   assign_negative=False)
        return cropped_image, labels, bboxes, distort_bbox

def preprocess_for_train(image, labels, bboxes,
                         out_shape, 
                         color_space='rgb', image_whitened=False, color_distort=False,
                         data_format='NHWC',
                         scope='ssd_preprocessing_train'):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.

    Returns:
        A preprocessed image.
    """
    fast_mode = False
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # bgr -> rgb
        # image = tf.reshape(image, [544,960,3])
        # image = tf.reshape(image, [1080,1920,3])
        # channels = tf.unstack(image, axis=-1)
        # image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

        # 0. Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
        # 1. rgb -> hsv
        if color_space == 'hsv':
            image = tf.image.rgb_to_hsv(image)
        # tf_summary_image(image, bboxes, 'image_with_bboxes')

        # # Remove DontCare labels.
        # labels, bboxes = ssd_common.tf_bboxes_filter_labels(out_label,
        #                                                     labels,
        #                                                     bboxes)

        # 2. Distort image and bounding boxes.
        dst_image = image
        dst_image, labels, bboxes, distort_bbox = \
             distorted_bounding_box_crop(image, labels, bboxes,
                                         min_object_covered=MIN_OBJECT_COVERED,
                                         aspect_ratio_range=CROP_RATIO_RANGE)
        # Resize image to output size.
        dst_image = tf_image.resize_image(dst_image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)                               
        # tf_summary_image(dst_image, bboxes, 'image_shape_distorted')

        # Randomly flip the image horizontally.
        # dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)

        # Randomly distort the colors. There are 4 ways to do it.
        # dst_image = apply_with_random_selector(
        #         dst_image,
        #         lambda x, ordering: distort_color(x, ordering, fast_mode),
        #         num_cases=4)
        # tf_summary_image(dst_image, bboxes, 'image_color_distorted')

        # Rescale to basemodel input scale.
        image = dst_image * 255.
        # if image_whitened:
        #     image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes