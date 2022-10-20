#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2022. Markus Knauer, Maximilian Denninger, Rudolph Triebel     #
# All rights reserved.                                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 21-10-2022                                                             #
# Author: Markus Knauer                                                        #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://github.com/DLR-RM/RECALL                                    #
################################################################################

"""
Script to measure the time it takes to convert the CORE50 dataset to RESNet50 features
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import sys
import os
import time
import argparse

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from recall.model.feature_extraction_network.resnet50 import ResNet50
from recall.data_import.dataset_converter_features import DataSetConverter
from recall.data_import.core50_dataset import CORE50


def convert_to_features(images: np.ndarray, resnet50_checkpoint_path: str):
    image_shapes = tf.TensorShape([None, tf.Dimension(128), tf.Dimension(128), tf.Dimension(3)])
    iterator = tf.data.Iterator.from_structure(tf.float32, image_shapes)
    input = iterator.get_next()
    res_net_output = ResNet50(input_tensor=input, resize=True)
    saver = tf.train.Saver(allow_empty=True)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    def gen_images() -> np.ndarray:
        for image in images:
            yield image

    dataset = tf.data.Dataset.from_generator(gen_images, (tf.float32), output_shapes=(128, 128, 3))
    dataset = dataset.map(lambda color_img: DataSetConverter.preprocess_images(color_img),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    iterator_init_op = iterator.make_initializer(dataset)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, resnet50_checkpoint_path)
        sess.run(iterator_init_op)
        while True:
            try:
                features = sess.run(res_net_output)
            except tf.errors.OutOfRangeError:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert the full CORE50 dataset to features")
    parser.add_argument("--path_to_dataset", help="Path to the dataset location", required=True)
    args = parser.parse_args()

    core_50 = CORE50(root=args.path_to_dataset, preload=False, scenario="nc")
    resnet50_checkpoint_path = os.path.join(args.path_to_dataset, "resnet50.ckpt")

    # add training images
    all_images = []
    for current_index, sequence_res in enumerate(core_50):
        images, _, _, _ = sequence_res
        all_images.extend(images)
    print("Loaded {} training images".format(len(all_images)))

    # add validation images
    dataset = core_50.get_full_valid_set(reduced=True)
    for current_index, sequence_res in enumerate(dataset):
        (images, _, _), _ = sequence_res
        all_images.extend(images)
    print("Loaded {} validation and training images".format(len(all_images)))
    print("Finish reading all images")
    start_time = time.time()
    convert_to_features(all_images, resnet50_checkpoint_path)
    print("Convert to features took: {}s".format(time.time() - start_time))
