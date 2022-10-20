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
Class with loads the dataset into the TensorFlow pipeline
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
from typing import Tuple, Dict

import yaml

import tensorflow.compat.v1 as tf

from recall.data_import.dataset_base import DataSetBase
from recall.utility.config import Configuration


class DataSetLoader(DataSetBase):

    def __init__(self, configuration: Configuration):
        """

        :param configuration: from config.py
        """

        DataSetBase.__init__(self, configuration)
        self._load_validation_set_once = True
        self._loaded_validation_ds: Dict[str, Tuple[tf.data.Dataset, tf.data.Dataset]] = {}

        self._iterator = self.get_current_iterator_structure()
        self._input, self._true_labels, self._rec_labels, self._input_path = self._iterator.get_next()

    def get_input(self):
        return self._input

    def get_iterator(self):
        return self._iterator

    def get_input_values(self):
        return self._true_labels, self._rec_labels, self._input_path

    @staticmethod
    def get_loader_iterator():
        return tf.data.Iterator.from_structure((tf.float32, tf.float32, tf.string),
                                               (tf.TensorShape([None, 2048]),  # features
                                                tf.TensorShape([None, None]),  # reconstruction label: batch
                                                                               # and class amount unknown
                                                tf.TensorShape([None])))

    def get_data_collection(self, sequence, mode):
        used_file_path = self._configuration.get_path_for_collection(sequence, mode)
        if not os.path.exists(used_file_path):
            raise FileNotFoundError(f"The file does not exist: {used_file_path}, maybe check if the generator "
                                    f"script created all!")
        with open(used_file_path, "r") as file:
            output = yaml.load(file, Loader=yaml.Loader)
        return output


    def get_tf_dataset(self, sequence: str, mode: str, shuffle: bool = True, repeat: bool = True) -> tf.data.Dataset:
        """
        Returns the tf.data.Dataset for the asked name, sequence and mode

        :param sequence: base, seq1-seqX
        :param mode: train, test, validation
        :param shuffle: If the dataset should be randomly shuffled
        :param repeat: If the dataset should be repeated
        :return: tf.dataset
        """
        if mode == "validation" and self._load_validation_set_once and sequence in self._loaded_validation_ds:
            return self._loaded_validation_ds[sequence]

        used_file_path = self._configuration.get_path_for(sequence, mode)
        if os.path.exists(used_file_path):
            def deserialize_tfrecord_features(example_proto) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                """
                Creates an TF Dataset from tf.Example message
                """
                keys_to_features = {'feature': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
                                    'label': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
                                    'path': tf.FixedLenFeature([], tf.string)}

                parsed_features = tf.parse_single_example(example_proto, keys_to_features)
                image = tf.cast(parsed_features['feature'], tf.float32)
                label = tf.cast(parsed_features['label'], tf.float32)
                path = tf.cast(parsed_features['path'], tf.string)

                return image, label, path

            autotune = tf.data.experimental.AUTOTUNE
            dataset = tf.data.TFRecordDataset(used_file_path, compression_type='GZIP')
            dataset = dataset.map(deserialize_tfrecord_features)
            dataset = dataset.cache()
            prev_dataset = dataset
            # add a fake rec label
            dataset = dataset.map(lambda color_img, label, path: (color_img, label, label, path),
                                  num_parallel_calls=autotune)
            dataset = self.finalize_dataset(dataset, repeat, shuffle)
            if mode == "validation" and self._load_validation_set_once:
                self._loaded_validation_ds[sequence] = (dataset, prev_dataset)

            return dataset, prev_dataset

        else:
            raise FileNotFoundError(f"The file does not exist: {used_file_path}, maybe check if the generator "
                                    f"script created all!")

    def finalize_dataset(self, dataset: tf.data.Dataset, repeat: bool, shuffle: bool) -> tf.data.Dataset:
        """
        Finalize the dataset by adding batching, repeating and shuffling.

        :param dataset: The current used tf.data.Dataset
        :param repeat: If the dataset should be repeated
        :param shuffle: If the dataset should be randomly shuffled
        :return: The changed tf dataset
        """
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(self._configuration.shuffle_size)
        dataset = dataset.batch(self._configuration.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
