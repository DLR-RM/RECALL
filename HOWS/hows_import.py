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
Script to load HOWS images according to the continual learning order.
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
from os.path import dirname, join
from pathlib import Path
import h5py

class HOWSImport(object):

    def __init__(self, root_file, nr_prev_classes=None):
        """
        Get pictures of a specific sequence and training or validation
        :param root_folder: filename of the text file where the picture-paths are in. For example training_0.txt
        :param nr_prev_classes: cumulated number of classes before the current sequence
        """
        self.images = np.array([])
        self.labels = []
        self.nr_classes = None
        self.nr_images = None
        self.paths = []
        self.class_mapping = {
            "apple": 0, "ball": 1, "bowl": 2, "camera": 3, "cap": 4, "egg": 5, "glass_bottle": 6, "headset": 7,
            "milk": 8,
            "mug": 9, "pear": 10, "scissors": 11, "teddy": 12, "bag": 13, "banana": 14, "bread": 15, "can": 16,
            "computer_keyboard": 17, "fork": 18, "glasses": 19, "knife": 20, "mobile_phone": 21, "pan": 22, "pen": 23,
            "spoon": 24
        }

        with open(root_file, "r") as f:
            self.paths.extend(f.read().split("\n"))

        self.paths = list(filter(None, self.paths))  ##checks if there is a empty string in the list and removes it

        self.nr_images = len(self.paths)
        self.images = [self.load_image(join(str(Path(root_file).parent.parent), filename)) for filename in self.paths]

        class_names = []
        for filename in self.paths:
            name = os.path.basename(dirname(dirname(dirname(filename))))
            if name == "teddy_fix":
                class_names.append("teddy")
            else:
                class_names.append(name)

        for obj in class_names:
            for key, value in self.class_mapping.items():
                if obj.startswith(key):
                    self.labels.append(value)
                    break

        self.images = np.array(self.images)
        self.labels = np.array(self.labels).reshape(-1).astype(np.int)
        self.paths = np.array(self.paths)

        if nr_prev_classes is None:
            self.nr_classes = len(np.unique(self.labels))

        else:
            self.nr_classes = np.count_nonzero(np.unique(self.labels)) + nr_prev_classes

        self.labels = np.eye(self.nr_classes)[self.labels].astype(np.float32)

    def get_X(self, max_amount=None):
        if max_amount is None:
            return self.images
        else:
            return self.images[:max_amount]

    def get_Y(self, max_amount=None):
        if max_amount is None:
            return self.labels
        else:
            return self.labels[:max_amount]

    def get_paths(self, max_amount=None):
        if max_amount is None:
            return self.paths
        else:
            return self.paths[:max_amount]

    def load_image(self, path):
        """
        Loads image from a given .hdf5 container path
        :param path: path to .hdf5 container
        :return: RGB image
        """
        if os.path.exists(path):
            if os.path.isfile(path):
                if path.endswith(".hdf5"):
                    with h5py.File(path, 'r') as data:
                        img = np.array(data["colors"], dtype=np.float32)
                    return img

        else:
            raise Exception(f"Path {path} doesnt exist!")


if __name__ == "__main__":

    import argparse
    import numpy as np
    from recall.utility.config import Configuration
    import logging
    from recall.utility.logger import logger

    parser = argparse.ArgumentParser("Loads images and labels of HOWS for a given sequence")
    parser.add_argument("--path_to_datasets", help="Path to the dataset location", required=True)
    parser.add_argument("--dataset_name", help="Name of the dataset (hows or hows_long)", required=True,
                        choices=["hows", "hows_long"], default="hows")
    parser.add_argument("--sequence", help="Sequence of the dataset (base, seq1, ..., seqX)")
    parser.add_argument("--mode", help="Mode of the dataset (train, validation)", required=True)
    args = parser.parse_args()

    config = Configuration(args.path_to_datasets, args.dataset_name)
    main_path = config.original_path_to_dataset
    logging.getLogger().setLevel(logging.WARNING)

    if args.sequence is None:
        seq_logger_output = "all"
    else:
        seq_logger_output = args.sequence
    logger().info(
        "Started DataSetConverterFeatures for Dataset {}, Sequence {}, Mode {}".format(args.dataset_name, seq_logger_output,
                                                                                       args.mode))

    if args.dataset_name == "hows_long":
        sequence_case = "_long_version"
    else:
        sequence_case = ""

    final_path = os.path.join(main_path, "HOWS_CL_25", "Sequences")
    if args.sequence == "base":
        if args.mode == "train":
            final_path = final_path + "/training_0{}.txt".format(sequence_case)
        elif args.mode == "validation":
            final_path = final_path + "/validation_0{}.txt".format(sequence_case)
        else:
            raise Exception("This combination does not exist!")
    elif "seq" in args.sequence:
        nr = int(args.sequence[3:])
        if args.mode == "train":
            final_path = final_path + "/training_{}{}.txt".format(nr, sequence_case)
        elif args.mode == "validation":
            final_path = final_path + "/validation_{}{}.txt".format(nr, sequence_case)
        else:
            raise Exception("This combination does not exist!")
    else:
        raise Exception("This sequence mode does not exist!")

    if os.path.exists(final_path):
        if args.sequence == "base":
            nr_prev_classes = None
        else:
            seq_nr = int(args.sequence[3:])
            if args.special_sequence_case:
                nr_prev_classes = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23][seq_nr]  # long version of HOWS
            else:
                nr_prev_classes = [0, 5, 10, 15, 20][seq_nr]
        image_provider = HOWSImport(final_path, nr_prev_classes)
    else:
        raise Exception("The created path does not exist: {}".format(final_path))

    images, labels, paths = image_provider.get_X(), image_provider.get_Y(), image_provider.get_paths()

    print("Loading of images successful, but you did not specify how to proceed further. Please do so by adapting the "
          "code of HOWS/hows_import.py at the end!")
    # From here you can hand the information over to your approach e.g. myapproach(images, labels) or to create
    # features out of them e.g. convertToFeatures(images, labels, paths).
