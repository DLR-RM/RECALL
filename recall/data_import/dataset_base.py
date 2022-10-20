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

Base Class for all Datasets

"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import tensorflow.compat.v1 as tf

from recall.utility.config import Configuration


class DataSetBase:

    def __init__(self, configuration: Configuration):
        """

        :param configuration: from config.py
        """
        self._configuration: Configuration = configuration


    @staticmethod
    def get_current_iterator_structure():
        """
        returns iterator in case of using features instead of images
        """
        iterator = tf.data.Iterator.from_structure((tf.float32, tf.float32, tf.float32, tf.string), (
            tf.TensorShape([None, 2048]),  # features from ResNet50
            tf.TensorShape([None, None]),  # true labels
            tf.TensorShape([None, None]),  # reconstruction label: batch and class amount unknown
            tf.TensorShape([None])))  # path
        return iterator

