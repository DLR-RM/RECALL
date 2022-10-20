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
Script to download and prepare the datasets CORe50 and iCIFAR-100 as .tfrecords for continual learning.
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
import argparse
import subprocess

from recall.utility.config import Configuration

def download_core50_dataset(config: Configuration):
    download_paths = ["https://vlomonaco.github.io/core50/data/labels.pkl",
                      "https://vlomonaco.github.io/core50/data/LUP.pkl",
                      "https://vlomonaco.github.io/core50/data/paths.pkl",
                      "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"]
    path_to_dataset = config.dataset_folder_path
    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)
    for path in download_paths:
        file_path = os.path.join(path_to_dataset, path[path.rfind("/") + 1:])
        print("Check {}".format(file_path))
        if not os.path.exists(file_path) and not os.path.exists(file_path.replace(".zip", "")):
            download_core50_cmd = "wget {} -P {}".format(path, path_to_dataset)
            print("Executing the following command: {}".format(download_core50_cmd))
            subprocess.call(download_core50_cmd, shell=True)
    core50_128x128_name = "core50_128x128"
    core_zip = os.path.join(path_to_dataset, core50_128x128_name + ".zip")
    if os.path.exists(core_zip):
        subprocess.call("unzip {}".format(core50_128x128_name), shell=True, cwd=path_to_dataset)
    # remove the .zip file if it is already unpacked
    if os.path.exists(os.path.join(path_to_dataset, core50_128x128_name)):
        if os.path.exists(core_zip):
            os.remove(core_zip)
    else:
        raise Exception("Something did not work with unzip {}, check if that file exists".format(core_zip))


def download_resnet50_weights(path_to_datasets: str):
    """
    Downloads the and saves weights of ResNet50
    :param path_to_datasets: Where the datasets are stored, default is Recall_datasets
    """
    if not os.path.exists(os.path.join(path_to_datasets, "resnet50.ckpt.index")):
        download_res_net_cmd = "python {} --path_to_dataset {}".format(os.path.join(os.path.dirname(__file__),
                                                                                    "save_resnet_weights.py"),
                                                                       path_to_datasets)
        subprocess.call(download_res_net_cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert CORE50 or iCifar100 dataset to .tfrecords, this will also download "
                                     "the dataset if it does not exist yet.")
    parser.add_argument("--path_to_datasets", help="Path to the dataset location", required=True)
    parser.add_argument("--dataset_name", help="Name of the used dataset", required=True,
                        choices=["core50", "icifar100"])
    args = parser.parse_args()

    config = Configuration(args.path_to_datasets, name_of_the_dataset=args.dataset_name)

    if args.dataset_name == "core50":
        download_core50_dataset(config)

    download_resnet50_weights(config.original_path_to_dataset)

    amount_of_sequences = 10
    python_script = os.path.join(os.path.dirname(__file__), "dataset_converter_features.py")
    for i in range(amount_of_sequences):
        for mode in ["validation", "train"]:
            cmd = f"python {python_script} --path_to_dataset {args.path_to_datasets} " \
                  f"--dataset_name {args.dataset_name} --mode {mode} --sequence {i}"
            subprocess.call(cmd, shell=True)
