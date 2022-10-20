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
Script to download and save ResNet50 weights
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.keras import backend as K
from recall.model.feature_extraction_network.resnet50_keras import ResNet50


def create_resnet_weights(path_to_dataset):
    """
    Only for loading the weights of ResNet50, using imagenet, and saving them in a .ckpt file (for the model to laod)
    """
    # load old resnet
    ResNet50(include_top=False, weights="imagenet", pooling="avg")
    sess = K.get_session()
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(path_to_dataset, "resnet50.ckpt"))
    print("Saved weights of ResNet50")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create resnet weights and store them in the given folder")
    parser.add_argument("--path_to_dataset", help="Path in which the resenet checkpoint will be stored", required=True)
    args = parser.parse_args()

    create_resnet_weights(args.path_to_dataset)
