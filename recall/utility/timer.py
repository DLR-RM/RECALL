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
Timer to measure how long training takes
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"
import time

from utility.TimeFrame import TimeFrame
from utility.Logger import logger

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


class NamedTimer(object):

    def __init__(self, name, frame=False):
        self._name = name
        self._start = None
        self._frame = frame
        if self._frame:
            logger().info("###############################################")

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args):
        elapsed_time = time.time() - self._start
        logger().info("'{}' took: {} to finish".format(self._name, TimeFrame(elapsed_time)))
        if self._frame:
            logger().info("###############################################")





if __name__ == "__main__":

    with NamedTimer("test_1"):
        time.sleep(1)
        print("Do something important")

    with NamedTimer("test_2", frame=True):
        time.sleep(1)
        print("Do something very important")

