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
Script which converts images to .tfrecords.
Fore CORE50 and iCIFAR-100 dataset.
This will also download the dataset if it does not exist yet.
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
from typing import List

import yaml
import logging
import sys
import pickle
import argparse
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

tf.disable_v2_behavior()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from recall.data_import.cifar100_dataset import CifarDataset
from recall.data_import.dataset_base import DataSetBase
from recall.model.feature_extraction_network.resnet50 import ResNet50
from recall.data_import.core50_dataset import CORE50
from recall.utility.config import Configuration


class DataSetConverter(DataSetBase):

    def __init__(self, configuration):
        """

        :param configuration: from config.py
        """
        DataSetBase.__init__(self, configuration)

    @staticmethod
    def preprocess_images(img: tf.Tensor) -> tf.Tensor:
        """
        Resnet50 is using mode 'caffe': caffe will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
        """
        img = img[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        img = img - mean
        # tf.dtypes.cast(img, tf.int32)
        return img

    @staticmethod
    def use_augmentation(img: tf.Tensor, cropping: bool = True) -> tf.Tensor:
        color_img = tf.image.random_hue(img, 0.12)
        color_img = tf.image.random_saturation(color_img, 0.44, 1.6)
        color_img = tf.image.random_brightness(color_img, 0.1)
        color_img = tf.image.random_contrast(color_img, 0.5, 1.5)
        # add flipping, cropping, scaling, more you can find online
        if cropping:
            color_img = tf.image.random_crop(color_img, [128, 128, 3])
        color_img = tf.image.random_flip_left_right(color_img)
        color_img = tf.image.random_flip_up_down(color_img)
        random_angles = tf.random.uniform(shape=(1,), minval=-np.pi / 4, maxval=np.pi / 4)

        color_img = tfa.image.transform(color_img, tfa.image.angles_to_projective_transforms(
            random_angles, tf.cast(tf.shape(color_img)[0], tf.float32), tf.cast(tf.shape(color_img)[1], tf.float32)),
                                        interpolation="BILINEAR")
        return color_img

    @staticmethod
    def serialize_example(feature, label, path):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature_output = {'feature': _float_feature(feature), 'label': _float_feature(label),
                          'path': _bytes_feature(str(path).encode('utf-8'))}

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature_output))
        return example_proto.SerializeToString()

    @staticmethod
    def write_to_tf_record(tf_path: str, features: List[np.ndarray], labels: List[np.ndarray], paths: List[str]):

        # check if folder for tf record file exists
        tf_folder = os.path.dirname(tf_path)
        if not os.path.exists(tf_folder):
            os.makedirs(tf_folder)

        counter = 0
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(tf_path, options=options)
        print(f"Start with serializing, final len: {len(features)}")
        config = tf.ConfigProto(device_count={'GPU': 0})
        with tf.Session(config=config) as sess:
            for feature, label, path in zip(features, labels, paths):
                res_ser = DataSetConverter.serialize_example(feature, label, path)
                writer.write(res_ser)
                counter += 1
                if counter % 100 == 0:
                    print("Image counter: {}".format(counter))
        writer.close()


    def _create_data_collection_info(self, images, labels, sequence, mode):
        """

        :param images: [nr_of_images, res_x, res_y, channels]
        :param labels: [nr_of_images, nr_of_classes]
        :param sequence: base, seq1-seqX
        :param mode: train, validation
        """
        file_path = self._configuration.get_path_for_collection(sequence, mode)

        # labels not in one hot encoding
        arg_maxed_labels = np.argmax(labels, axis=1).astype(int)

        # counts all elements
        unique, counts = np.unique(arg_maxed_labels, return_counts=True)
        classes = dict(zip(unique.tolist(), counts.tolist()))

        nr_classes = len(unique)
        max_nr_classes = int(np.max(arg_maxed_labels)) + 1

        data_dict = {
            "nr_images": images.shape[0],
            "nr_classes": nr_classes,
            "x.shape": list(images.shape[1:]),
            "y.shape": [max_nr_classes],
            "classes": classes  # {Class_id: Nr of img per class}
        }

        with open(file_path, 'w') as f:
            f.write(yaml.dump(data_dict))

    def convert_to_tf_record(self, images, labels, paths, sequence, mode):
        """

        :param images: [nr_of_images, res_x, res_y, channels]
        :param labels: [nr_of_images, nr_of_classes]
        :param paths: to identify each picture
        :param sequence: base, seq1-seqX
        :param mode: train, test, validation
        """

        file_path = self._configuration.get_path_for(sequence, mode)

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        print("Total amount of images: {} and labels: {}".format(images.shape[0], labels.shape[0]))
        self._create_data_collection_info(images, labels, sequence, mode)
        labels = labels.astype(np.float32)

        number_of_epochs_per_set = self._configuration.number_of_epochs_per_set(mode=mode)
        list_of_pickle_files = []

        image_shapes = tf.TensorShape([None, tf.Dimension(128), tf.Dimension(128), tf.Dimension(3)])
        iterator = tf.data.Iterator.from_structure((tf.float32, tf.float32, tf.string),
                                                   (image_shapes, tf.TensorShape([None, None]),
                                                    tf.TensorShape([None])))

        input, output_labels, output_paths = iterator.get_next()
        # create ResNet
        res_net_output = ResNet50(input_tensor=input, resize=True)

        saver = tf.train.Saver(allow_empty=True)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            res_net_data = (input, output_labels, output_paths, res_net_output, iterator, sess)
            # init all weights
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self._configuration.resnet50_checkpoint_path)
            nr_of_files = 0
            for epoch_nr in range(number_of_epochs_per_set):
                self.feature_calculation(images, labels, paths, nr_of_files,
                                         epoch_nr == 0, res_net_data)
                new_pickle_file = os.path.join(self._configuration.general_path, "temp_folder",
                                               "data_{}.pickle".format(len(list_of_pickle_files)))
                nr_of_files += 1
                list_of_pickle_files.append(new_pickle_file)

        final_features = []
        final_labels = []
        final_paths = []
        for pickle_file in list_of_pickle_files:
            with open(pickle_file, "rb") as file:
                current_features, current_labels, current_paths = pickle.load(file)
                final_features.extend(current_features)
                final_labels.extend(current_labels)
                final_paths.extend(current_paths)
            os.remove(pickle_file)
        DataSetConverter.write_to_tf_record(file_path, final_features, final_labels, final_paths)

    def feature_calculation(self, images, labels, paths, nr_of_files, is_first_epoch, res_net_data):
        input, output_labels, output_paths, res_net_output, iterator, sess = res_net_data

        cropping = True
        autotune = tf.data.experimental.AUTOTUNE

        def gen_images():
            for image, label, path in zip(images, labels, paths):
                yield image, label, path

        dataset = tf.data.Dataset.from_generator(gen_images, (tf.float32, tf.float32, tf.string))
        if not is_first_epoch:
            dataset = dataset.map(
                lambda color_img, label, path: (DataSetConverter.use_augmentation(color_img, cropping),
                                                label, path), num_parallel_calls=autotune)
        dataset = dataset.map(
            lambda color_img, label, path: (DataSetConverter.preprocess_images(color_img), label, path),
            num_parallel_calls=autotune)
        dataset = dataset.batch(128)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        iterator_init_op = iterator.make_initializer(dataset)

        print("Start predicting")
        sess.run(iterator_init_op)
        inner_counter = 0
        current_features = []
        current_labels = []
        current_paths = []
        while True:
            try:
                features, true_labels, used_paths = sess.run([res_net_output, output_labels, output_paths])
                current_features.extend(features)
                current_labels.extend(true_labels)
                current_paths.extend(used_paths)
                inner_counter += 1
            except tf.errors.OutOfRangeError:
                print("Reach end of dataset")
                break
        new_pickle_file = os.path.join(self._configuration.general_path, "temp_folder",
                                       "data_{}.pickle".format(nr_of_files))
        if not os.path.exists(os.path.dirname(new_pickle_file)):
            os.makedirs(os.path.dirname(new_pickle_file))
        with open(new_pickle_file, "wb") as file:
            pickle.dump((current_features, current_labels, current_paths), file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Convert CORE50 and iCIFAR-100 dataset to .tfrecords, this will also download the dataset "
                                     "if it does not exist yet.")
    parser.add_argument("--path_to_dataset", help="Path to the dataset location", required=True)
    parser.add_argument("--dataset_name", help="Name of the used dataset", required=True,
                        choices=["core50", "icifar100"])
    parser.add_argument("--mode", help="Mode either train or validation", required=True)
    parser.add_argument("--sequence", help="Sequence which should be used", type=int, required=True)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)

    config = Configuration(args.path_to_dataset, name_of_the_dataset=args.dataset_name)
    config.resnet50_checkpoint_path = os.path.join(args.path_to_dataset, "resnet50.ckpt")

    data_set_converter = DataSetConverter(config)

    mode = args.mode
    sequence = args.sequence

    current_seq = "seq{}".format(sequence)
    if sequence == 0:
        current_seq = "base"
    print("Current sequence: {}".format(current_seq))

    if args.dataset_name == "core50":
        core_50 = CORE50(root=args.path_to_dataset, preload=False, scenario="nc")

        if mode == "train":
            dataset = core_50
        elif mode == "validation":
            dataset = core_50.get_full_valid_set(reduced=True)
        else:
            raise ValueError("This mode is unknown: {}".format(mode))
        total_class_max = 0
        for current_index, sequence_res in enumerate(dataset):
            if current_index < sequence:
                continue
            if mode == "train":
                images, labels, paths, _ = sequence_res
            else:
                (images, labels, paths), _ = sequence_res

            labels = labels.astype(int)

            print("images {}".format(images.shape))
            print("labels {}".format(labels.shape))

            nr_classes = np.max(labels) + 1
            nr_classes = np.max([total_class_max, nr_classes])
            labels = np.eye(nr_classes)[labels].astype(np.float32)
            total_class_max = nr_classes

            data_set_converter.convert_to_tf_record(images, labels, paths, current_seq, mode)
            break
    elif args.dataset_name == "icifar100":
        icifar = CifarDataset(sequence=current_seq, mode=args.mode)
        data_set_converter.convert_to_tf_record(icifar.images, icifar.labels, icifar.paths, current_seq, mode)
    else:
        raise ValueError("The dataset is unknown!")

