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
Script to download the full HOWS_CL_25 dataset (RGB-D, normal and segmenation images).
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
import argparse
import subprocess
from pathlib import Path

from recall.utility.config import Configuration


def download_hows_features(path_to_dataset: str) -> str:
    download_path = "https://zenodo.org/record/7189434/files/HOWS_CL_25_hdf5_features.zip"

    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)

    output_file = "HOWS_CL_25_hdf5_features.zip"
    final_path = os.path.join(path_to_dataset, output_file)
    if not os.path.exists(final_path):
        cmd = f"curl {download_path} --output {final_path}"
        subprocess.run(cmd, shell=True)
    if os.path.exists(final_path):
        # unzip the final file
        subprocess.run("unzip {} {}".format(final_path, path_to_dataset))
    else:
        raise RuntimeError("The download failed the file was not correctly downloaded!")
    # remove the zip file
    os.remove(final_path)

    return final_path[:-len(".zip")]


def download_hows_images(path_to_dataset: str):
    # download the files first
    download_links = ["https://zenodo.org/record/7189434/files/HOWS_CL_25.zip",
                      "https://zenodo.org/record/7189434/files/HOWS_CL_25.z01",
                      "https://zenodo.org/record/7189434/files/HOWS_CL_25.z02"]
    for download_link in download_links:
        subprocess.run("wget {} -P {}/".format(download_link, path_to_dataset), shell=True)
    part_files = [os.path.join(path_to_dataset, "HOWS_CL_25" + ending) for ending in [".zip", ".z01", ".z02"]]
    for part_file in part_files:
        if not os.path.exists(part_file):
            raise FileNotFoundError("The download of the part file \"{}\" failed".format(part_file))

    final_file = os.path.join(path_to_dataset, "hows_cl_25.zip")
    subprocess.run("cat {} > {}".format(os.path.join(path_to_dataset, "HOWS_CL_25.z*"), final_file), shell=True)

    # unzip the final file
    subprocess.run("unzip {} {}".format(final_file, path_to_dataset))

    hows_folder = os.path.join(path_to_dataset, "HOWS_CL_25")
    if os.path.exists(hows_folder):
        for part_file in part_files:
            os.remove(part_file)
        os.remove(final_file)
        print("The download is completed and the extraction happened: {}".format(hows_folder))
    else:
        raise FileNotFoundError("Something went wrong the final HOWS folder is not there!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Download script to get the HOWS-CL-25 dataset.")
    parser.add_argument("--path_to_datasets", help="Folder path in which the final HOWS dataset will be stored",
                        required=True)
    parser.add_argument("--images", help="Download all the images", default=False, action="store_true")
    parser.add_argument("--features", help="Download the ResNet50 features", default=False, action="store_true")
    parser.add_argument("--convert_to_tfrecord", help="Coverts the downloaded .hdf5 files to .tfrecord", default=False,
                        action="store_true")
    args = parser.parse_args()

    config = Configuration(args.path_to_datasets, name_of_the_dataset="hows")
    path_to_dataset = config.dataset_folder_path
    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)

    if args.images:
        download_hows_images(path_to_dataset)

    if args.features:
        download_hows_features(path_to_dataset)
        if args.convert_to_tfrecord:
            cwd = Path(__file__).parent.parent.absolute()
            subprocess.run(f"python HOWS/create_tf_features.py --path_to_dataset {args.path_to_datasets} "
                           f"--dataset_name hows", shell=True, cwd=str(cwd))
            subprocess.run(f"python HOWS/create_tf_features.py --path_to_dataset {args.path_to_datasets} "
                           f"--dataset_name hows_long", shell=True, cwd=str(cwd))