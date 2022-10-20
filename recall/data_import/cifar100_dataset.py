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
Class for iCIFAR-100 Dataset
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import tensorflow.compat.v1 as tf
import numpy as np
import cv2


class CifarDataset:

    def __init__(self, sequence: str, mode: str):
        """
        :param sequence: The current sequence name
        :param mode: train, validation
        """

        nr_of_classes_per_sequence = 10
        if sequence == "base":
            nr_prev_classes = 0
        else:
            nr_prev_classes = int(sequence[len("seq"):]) * nr_of_classes_per_sequence

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        if mode == "train":
            self.images = x_train
            self.labels = y_train
        else:
            self.images = x_test
            self.labels = y_test

        self.paths = np.array([""] * len(self.images))

        sorted_indices = np.array([i for _, i in sorted(zip(self.labels, np.arange(len(self.labels))), key=lambda x: x[0])], dtype=np.int32)
        self.images = np.array(self.images)[sorted_indices]
        self.labels = np.array(self.labels)[sorted_indices]

        # delete pictures which are not used in the current sequence
        first_class = nr_prev_classes
        current_class_range = first_class + nr_of_classes_per_sequence
        not_deleted_indices = self.labels >= first_class
        not_deleted_indices = np.bitwise_and(self.labels < current_class_range, not_deleted_indices)[:, 0]
        self.labels = self.labels[not_deleted_indices].copy()
        self.images = self.images[not_deleted_indices].copy()
        self.paths = self.paths[not_deleted_indices].copy()

        self.nr_images = len(self.images)
        self.labels = np.array(self.labels).reshape(-1).astype(np.int)

        self.nr_classes = nr_of_classes_per_sequence + nr_prev_classes

        # resize image to 128 x 128
        self.images = np.array([cv2.resize(image, (128, 128)) for image in self.images])

        self.images = np.array(self.images).astype(np.float32)
        print("self.Number of classes is {}".format(self.nr_classes))
        print("nr_prev_classes: {}".format(nr_prev_classes))
        print("images are of shape: {}".format(self.images.shape))
        print("labels are of shape: {}".format(self.labels.shape))
        self.labels = np.eye(self.nr_classes)[self.labels].astype(np.float32)
