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
Reads the configuration from the config.yaml
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
from pathlib import Path
from datetime import datetime
from typing import List

import yaml

from recall.utility.logger import logger, Logger


class Configuration(object):

    def __init__(self, path_to_dataset: str, name_of_the_dataset: str,
                 use_divide_by_variance: bool = False, use_regression_loss: bool = False,
                 record_tensorboard_session: bool = False):
        self.database = name_of_the_dataset
        # find the current run type
        if not use_divide_by_variance and not use_regression_loss:
            current_run_name = "default"
        elif use_divide_by_variance and not use_regression_loss:
            current_run_name = "divide_by_variance"
        elif not use_divide_by_variance and use_regression_loss:
            current_run_name = "regression_loss"
        elif use_divide_by_variance and use_regression_loss:
            current_run_name = "divide_by_variance_with_regression"
        else:
            raise ValueError("Unknown run config!")
        print(f"This is a \"{current_run_name.replace('_', ' ')}\" run!")

        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError("The config.yaml file could not be found, was it deleted?")
        # extract loaded values
        with open(config_path, "r") as file:
            data_dict = yaml.load(file, Loader=yaml.Loader)

        current_data_dict = data_dict["recall"][current_run_name][self.database]

        # set the config values
        self.do_divide_with_var = use_divide_by_variance
        self.using_reconstruction_loss = use_regression_loss
        self.seq_learning_rate = current_data_dict["seq_learning_rate"]
        self.siren_omega = current_data_dict["siren_omega"]
        self.base_steps = current_data_dict["base_steps"]
        self.seq_steps = current_data_dict["seq_steps"]
        self.initialize_heads = current_data_dict["initialize_heads"]
        self.take_weights_from_base_head = current_data_dict["take_weights_from_base_head"]
        self.use_combined_loss = current_data_dict["use_combined_loss"]

        # set fixed config values
        self.batch_size = 512
        self.shuffle_size = 32768
        self.record_tensorboard = record_tensorboard_session

        if self.is_core50():
            self.dataset_folder_name = "CORe50"
        elif self.is_hows():
            self.dataset_folder_name = "HOWS"
        elif self.is_hows_long():
            self.dataset_folder_name = "HOWS_LONG"
        elif self.is_icifar():
            self.dataset_folder_name = "iCifar100"
        else:
            raise ValueError("Unknown dataset was provided!")

        self.original_path_to_dataset = path_to_dataset
        self.dataset_folder_path = os.path.join(path_to_dataset, self.dataset_folder_name)
        self.general_path = os.path.join(self.dataset_folder_path, "tf_records")
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
        self.tensorboard_path: Path = Path(__file__).parent.parent.parent / "logs" / current_time
        if not self.tensorboard_path.exists():
            os.makedirs(self.tensorboard_path)

        log_file_path = self.tensorboard_path / "log_file.log"
        Logger.init(log_file_path)
        logger().log_file("Created log file for run with dataset: {}".format(self.database))
        self.save_config_to_logs()

    def save_config_to_logs(self):
        config_file_path_save = self.tensorboard_path / "config_save.yaml"
        object_dir = self.__dict__
        with config_file_path_save.open('w') as f:
            f.write(yaml.dump(object_dir))

    def is_core50(self):
        return self.database.lower() == "core50"

    def is_hows(self):
        return self.database.lower() == "hows"

    def is_hows_long(self):
        return self.database.lower() == "hows_long"

    def is_icifar(self):
        return self.database.lower() == "icifar100"

    def number_of_epochs_per_set(self, mode: str):
        if self.is_core50():
            if mode == "train":
                return 4
            else:
                return 1
        elif self.is_icifar():
            if mode == "train":
                return 2
            else:
                return 1
        else:
            raise NotImplemented("")

    def get_order_list(self) -> List[str]:
        """
        Get the list of sequences which should be executed one after the other
        Returns: list of strings
        """
        if self.is_core50():
            return ["base", "seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7", "seq8"]
        elif self.is_hows():
            return ["base", "seq1", "seq2", "seq3", "seq4"]
        elif self.is_hows_long():
            return ["base", "seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7", "seq8", "seq9",
                    "seq10", "seq11"]
        elif self.is_icifar():
            return ["base", "seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7", "seq8", "seq9"]
        else:
            raise ValueError("The order list for the current dataset is not specified")

    def get_path_for(self, sequence: str, mode: str) -> str:
        """
        Get the path for the tf record for this sequence and mode

        :param sequence: base, seq1-seqX
        :param mode: train, test, validation
        :return: the full path to the .tfrecord file
        """

        file_name = "resnet50_data.tfrecord"
        return os.path.join(self.general_path, sequence, mode, file_name)

    def get_path_for_collection(self, sequence: str, mode: str) -> str:
        """
        Get the path to the collection file for the sequence and mode

        :param sequence: base, seq1-seqX
        :param mode: train, test, validation
        :return: the full path to the collection.yaml file
        """
        return os.path.join(self.general_path, sequence, mode, "collection.yaml")
