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
Logs the RECALL training and saves it to /logs
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import logging
import sys


def logger():
    return Logger.get_logger()

class Logger(object):

    _logger = None

    @staticmethod
    def init(log_file_location):
        if Logger._logger is None:
            Logger._logger = logging.getLogger("own_logger")
            Logger._logger.propagate = False
            # set highest logger to let everything through
            logging.getLogger().setLevel(logging.NOTSET)
            # to let all messages through
            Logger._logger.setLevel(logging.NOTSET)


            formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(funcName)s:%(lineno)d: %(message)s",
                                      "%H:%M:%S")

            file_handler = logging.FileHandler(log_file_location)
            file_handler.setFormatter(formatter)
            # to get everything
            file_handler.setLevel(logging.NOTSET)

            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter("%(levelname)s:%(funcName)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.DEBUG)

            Logger._logger.addHandler(console_handler)
            Logger._logger.addHandler(file_handler)
            LOG_FILE_LEVEL = 8
            logging.addLevelName(LOG_FILE_LEVEL, "LOG_FILE")

            def log_file_level(self, message, *args, **kws):
                if self.isEnabledFor(LOG_FILE_LEVEL):
                    self._log(LOG_FILE_LEVEL, message, args, **kws)
            logging.Logger.log_file = log_file_level

        else:
            Logger._logger.warning("The Logger has already been inited!")

    @staticmethod
    def get_logger():
        if Logger._logger is not None:
            return Logger._logger
        else:
            print("ERROR: logger was not inited!")



if __name__ == "__main__":

    Logger.init("test.log")
    logger().info("Info")
    logger().log_file("Only in the log file")
    logger().error("Error")
    logger().error("Error 2")

