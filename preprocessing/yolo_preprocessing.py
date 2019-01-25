# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
###################################################
# ----------------------------------------------- #
# |           Data Augmentation                 | #
# |                                             | #
# | Details:                                    | #
# |  1. Def preprocess_for_train()              | #
# | image mat (inputsize,inputsize,3)           | #
# | boxes = [ymin, xmin, ymax, xmax]    (num,4) | #
# | label = label     (num,)                    | #
# ----------------------------------------------- #
###################################################
from __future__ import absolute_import
from __future__ import division

from enum import Enum, IntEnum
import tensorflow as tf
import tf_extended as tfe
from tensorflow.python.ops import control_flow_ops
from preprocessing import tf_image

# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.

# Some training pre-processing parameters.
# BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.
BBOX_CROP_OVERLAP = 0.8
# MIN_OBJECT_COVERED = 0.25
MIN_OBJECT_COVERED = 0.7
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary. Numpy version.

    Returns:
      Centered image.
    """
    img = np.copy(image)
    img += np.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(np.uint8)
    return img

def tf_summary_image(image, bboxes, name='image', unwhitened=False):
    """Add image with bounding boxes to summary.
    """
    # if unwhitened:
    #     image = tf_image_unwhitened(image)
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)

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

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, labels, bboxes,
                         out_shape, 
                         color_space='rgb', image_whitened=False, color_distort=False,
                         data_format='NHWC',
                         scope='yolo_preprocessing_train'):
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
    with tf.name_scope(scope, 'yolo_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # 0. Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)      
        tf_summary_image(image, bboxes, 'image_with_bboxes')

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
        tf_summary_image(dst_image, bboxes, 'image_shape_distorted')

        # Randomly flip the image horizontally.
        dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)

        # Randomly distort the colors. There are 4 ways to do it.
        # dst_image = apply_with_random_selector(
        #         dst_image,
        #         lambda x, ordering: distort_color(x, ordering, fast_mode),
        #         num_cases=4)
        # tf_summary_image(dst_image, bboxes, 'image_color_distorted')

        # Rescale to basemodel input scale.
        image = dst_image * 255.
        if image_whitened:
            image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes

def preprocess_for_detect(image, labels, bboxes,
                         out_shape, 
                         color_space='rgb', image_whitened=False, color_distort=False,
                         data_format='NHWC',
                         scope='ssd_preprocessing_detect'):
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        bbox_img = tf.constant([[0., 0., 1., 1.]])
        image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        
        return image

def preprocess_for_val(image, labels, bboxes,
                         out_shape, 
                         color_space='rgb', image_whitened=False, color_distort=False,
                         data_format='NHWC',
                         scope='ssd_preprocessing_val'):
    
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        bbox_img = tf.constant([[0., 0., 1., 1.]])
        image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
        
        return image
