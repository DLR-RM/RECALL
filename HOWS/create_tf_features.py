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
Script to create .tfrecord files from .hdf5 files for HOWS and HOWS long
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from recall.data_import.dataset_converter_features import DataSetConverter
from recall.utility.config import Configuration


def get_hdf5_content(path_to_dataset: str, sequence: str, mode: str, is_long_version: bool):
    folder_path = Path(path_to_dataset) / "HOWS_CL_25_hdf5_features"
    if not folder_path.exists():
        raise FileNotFoundError(f"The {folder_path.name} was not found in the {path_to_dataset}, please"
                                f"execute the download script beforehand")

    file_path = folder_path / "features_hdf5" / sequence / mode
    if not is_long_version:
        file_path = file_path / "resnet50_data.hdf5"
        collection_file_name = "collection.yaml"
    else:
        file_path = file_path / "resnet50_data_long_version.hdf5"
        collection_file_name = "collection_long_version.yaml"

    current_collection_file = file_path.parent / collection_file_name

    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist!")

    with h5py.File(file_path, "r") as file:
        return tuple(np.array(file[key]) for key in ["features", "labels", "path"]), current_collection_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates the tf record files for HOWS and HOWS long")
    parser.add_argument("--path_to_datasets", help="Path to the folder of the compressed dataset", required=True)
    parser.add_argument("--dataset_name", help="Name of the used dataset", required=True,
                        choices=["hows", "hows_long"])
    args = parser.parse_args()

    # load the config
    config = Configuration(args.path_to_datasets, args.dataset_name)

    dataset = DataSetConverter(config)

    order_list = config.get_order_list()

    for current_seq in order_list:
        for mode in ["train", "validation"]:
            tf_path = config.get_path_for(current_seq, mode)
            folder_path = str((Path(config.original_path_to_dataset) / "HOWS").absolute())
            (features, labels, paths), collection_path = get_hdf5_content(folder_path, current_seq,
                                                                          mode, config.is_hows_long())
            DataSetConverter.write_to_tf_record(tf_path, list(features), list(labels), list(paths))
            new_collection_file = Path(tf_path).parent / "collection.yaml"
            shutil.copyfile(collection_path, new_collection_file)





