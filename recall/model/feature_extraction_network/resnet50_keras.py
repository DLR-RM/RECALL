"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras_applications.imagenet_utils import _obtain_input_shape

import keras_applications.resnet50 as resnet50

import tensorflow.compat.v1 as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils


WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    resnet50.backend = backend
    resnet50.layers = layers
    resnet50.models = models
    resnet50.keras_utils = utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=128,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

##############
    # Edit Layers
#############
    
    x = layers.Lambda(lambda image: tf.image.resize_images(image, (224,224), method = tf.image.ResizeMethod.BICUBIC, align_corners = True, preserve_aspect_ratio = True))(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet50.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = resnet50.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = resnet50.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = resnet50.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = resnet50.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = resnet50.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = resnet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = resnet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        #x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
        #x = layers.Dense(classes, name='fc1000')(x)
        x = layers.Dense(2048, name='fc1')(x)
        x = layers.Dense(1024, name='fc2')(x)
        x = layers.Dense(512, name='fc3')(x)
        x = layers.Dense(1024, name='fc4')(x)
        x = layers.Dense(512, name='fc5')(x)
        x = layers.Dense(256, name='fc6')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = tf.keras.utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path, by_name=True)
        if backend.backend() == 'theano':
            tf.keras.utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
