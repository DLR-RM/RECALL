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
Main script for RECALL. You can start everything from here.
e.g.
python recall.py --path_to_dataset $PATH_TO_DATASET --dataset_name "hows"
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # we only support tf 1.15 at the moment

from recall.utility.config import Configuration
from recall.model.recall_model import RecallModel
from recall.model.trainings_manager import TrainingsManager
from recall.data_import.dataset_loader import DataSetLoader

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Runs the RECALL approach")
    parser.add_argument("--path_to_datasets", help="Path to the folder of the datasets", required=True)
    parser.add_argument("--dataset_name", help="Name of the used dataset", required=True,
                        choices=["core50", "hows", "hows_long", "icifar100"])
    parser.add_argument("--use_divide_by_variance", help="This activates the use divide by variance setting",
                        default=False, action="store_true")
    parser.add_argument("--use_regression_loss", help="This activates the use of the regression loss instead of "
                                                      "the classification loss", default=False, action="store_true")
    parser.add_argument("--record_tensorboard", help="Starts the recording of a tensorboard session",
                        default=False, action="store_true")
    args = parser.parse_args()

    # load the config
    config = Configuration(args.path_to_datasets, args.dataset_name,
                           args.use_divide_by_variance, args.use_regression_loss,
                           args.record_tensorboard)

    # load dataset
    dataset_loader = DataSetLoader(config)

    # build the recall cnn model
    model = RecallModel(dataset_loader, config)

    # set up the trainings manager and start the training
    trainings_manager = TrainingsManager(config, dataset_loader, model)
    trainings_manager.run()

    print("Experiment complete")
