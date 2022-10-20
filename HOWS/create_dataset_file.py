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
Script to create your own image-loading order from HOWS in form of a .txt file
 1. Define your sequences and which categories should be in them (as shown in the example below)
 2. Set the percentage of validation to training images
 3. Give the path to the images and the path to the destination of the new order.txt file.
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
from os.path import join
from typing import List

def write_file(data: List[str], sequence: str, mode: str):
    """
    Writes the image order into a .txt file
    :param data: list of paths to HOWS images
    :param sequence: name of the sequence
    :param mode: validation or training
    """
    if mode == "training":
        file_name = "training_{}{}.txt".format(sequence, seq_name)
    elif mode == "validation":
        file_name = "validation_{}{}.txt".format(sequence, seq_name)
    else:
        raise Exception("Mode {} not supported yet".format(mode))

    output_path = sequence_path + file_name

    with open(output_path, 'w') as f:
        f.write("\n".join(data))


def get_paths(sequence: List[str], nr: int, writing_mode: bool = True):
    """
    Gets the paths of images
    :param sequence: list of categories in the sequence
    :param nr: sequence number
    :param writing_mode: If true the paths are written into the file, if not not.
    """
    validation = []
    training = []
    repeat = True

    for obj in sequence:
        obj_path = join(image_path, obj)
        count_files = len(os.listdir(obj_path))
        count_valid_files = 0
        lst = os.listdir(obj_path)
        lst.sort()
        for instance_f in lst:
            instance_path = join(obj_path, instance_f)
            repeat = (count_valid_files / float(count_files) * 100) < validation_percentage
            for hash_file in os.listdir(instance_path):
                file_path = os.path.join(instance_path, hash_file)
                for picture in os.listdir(file_path):
                    if picture.endswith(".hdf5"):
                        picture_path = join(file_path, picture)
                        if os.path.exists(picture_path):
                            if repeat:
                                validation.append(os.path.join(file_path, picture))
                            else:
                                training.append(os.path.join(file_path, picture))
                        else:
                            raise FileNotFoundError("File {} does not exist!".format(picture_path))
            if repeat:
                count_valid_files += 1

        if writing_mode:
            write_file(training, nr, "training")
            write_file(validation, nr, "validation")

    return training, validation


if __name__ == "__main__":
    # Five sequences with each having five objects, each objects having about 6000 Images - As used in the RECALL Paper
    seq_name = "default"
    nr_sequences = 5
    seq_0 = ["apple", "ball", "bowl", "camera", "cap"]
    seq_1 = ["egg", "glass_bottle", "headset", "milk", "mug"]
    seq_2 = ["pear", "scissors", "teddy", "teddy_fix", "bag", "banana"]
    seq_3 = ["bread", "can", "computer_keyboard", "fork", "glasses"]
    seq_4 = ["knife", "mobile_phone", "pan", "pen", "spoon"]
    seqs = [seq_0, seq_1, seq_2, seq_3, seq_4]

    # ----- Distribution in validation and training
    # The first 10 % Instance of each object is for validation, each else for training
    validation_percentage = 10

    # ----- Path to images
    image_path = "/HOWS_CL_25/Images/"
    sequence_path = "/HOWS_CL_25/Sequences/"

    for i, seq in enumerate(seqs):
        get_paths(seq, i)

    print("Done")
