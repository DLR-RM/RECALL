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
Training Manager class to manage the continual learning pipeline
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import math
import time
import yaml

import numpy as np
import tensorflow.compat.v1 as tf

from recall.utility.logger import logger


def expand_label(label, diff_classes):
    label = tf.concat([label, tf.zeros(diff_classes, tf.float32)], 0)
    return label


class TrainingsManager(object):

    def __init__(self, configuration, dataset_loader, model):
        self._config = configuration
        self.order_of_sequences = self._config.get_order_list()
        self.perform_final_test = False
        self._dataset_loader = dataset_loader
        self._model = model
        self._sess = model._sess
        self._stats = {"disk": [], "ram": []}
        self._acc_dict = {}
        self._track_kernel_vars = []


    def run(self):
        start_time = time.time()
        # with rec and true label
        dataset_name = self._config.database

        collection_of_previous_sequences = []
        total_amount_of_classes = 0
        amount_of_classes_per_seq = {}
        list_of_validation_accs = []
        set_of_classes = None
        list_of_variances_per_sequence = []
        for current_sequence in self.order_of_sequences:
            logger().info("Start sequence: {}".format(current_sequence))
            collection_of_previous_sequences.append(current_sequence)

            sequence_mode = "base" not in current_sequence

            # get dataset
            should_repeat = not sequence_mode
            should_shuffle = not sequence_mode
            next_trainings_dataset, prev_dataset = self._dataset_loader.get_tf_dataset(sequence=current_sequence,
                                                                                       mode="train", repeat=should_repeat, shuffle=should_shuffle)
            next_data_collection = self._dataset_loader.get_data_collection(sequence=current_sequence, mode="train")

            old_amount_of_classes = total_amount_of_classes
            current_set_of_classes = set([class_nr for class_nr in next_data_collection["classes"].keys()])
            if set_of_classes == None:
                set_of_classes = current_set_of_classes
            else:
                set_of_classes = set_of_classes.union(current_set_of_classes)

            total_amount_of_classes = len(set_of_classes)
            amount_of_classes_per_seq[current_sequence] = total_amount_of_classes
            amount_of_images = next_data_collection["nr_images"]

            amount_of_new_classes = total_amount_of_classes - old_amount_of_classes

            # expand model
            if amount_of_new_classes > 0:
                self._model.add_new_classes(amount_of_new_classes)

            # change to new dataset
            self._model.change_to_dataset(next_trainings_dataset)

            if sequence_mode:

                # we have to predict the reconstruction labels and extend the CNN for the next training
                next_trainings_dataset, variance = self.add_reconstruction_labels_to_tf_dataset(prev_dataset, shuffle=True)
                # set the new variance to divide the values correctly
                self._model._variance_per_class.load(variance[:self._model.old_amount_of_cl], self._sess)
                # change to the newly created model
                self._model.change_to_dataset(next_trainings_dataset)

            # increase amount as we used augmentations during the generation
            amount_of_images *= 4
            steps_per_epoch = int(math.ceil(amount_of_images / float(self._config.batch_size)))

            if not sequence_mode:
                nr_training_steps = self._config.base_steps
            else:
                nr_training_steps = self._config.seq_steps

            if self._config.record_tensorboard:
                self._model.create_new_file_writer(current_sequence, "train")

            list_of_operations = self._model.get_list_of_ops_for("train")
            self._sess.run(tf.local_variables_initializer())
            global_counter = 0
            
            trainings_dict = self._model.get_current_batch_dict(False)

            for epoch_nr in range(nr_training_steps):
                for batch_nr in range(steps_per_epoch):
                    accuracy = self._model.perform_train_step(list_of_operations, global_counter, trainings_dict)
                    global_counter += 1
                    if batch_nr % 10 == 0:
                        logger().info("\tCurrent batch: {} of {}, has acc: {}".format(batch_nr + 1, steps_per_epoch, accuracy))
                logger().info("Done with epoch: {} of {}, has acc: {}".format(epoch_nr + 1, nr_training_steps, accuracy))
            if "seq1" in self._model._variable_collection:
                filtered_vars = [ele for ele in self._model._variable_collection["seq1"] if "kernel" in str(ele) and "dense_layer_before_softmax" in str(ele) and "seq_1" in str(ele)]
                self._track_kernel_vars.append(self._model._sess.run(filtered_vars[0]))

            logger().info("These are the values of the sums of the dense layer before softmax weights of the seq1" + str([np.sum(np.abs(ele)) for ele in self._track_kernel_vars]))

            # if you only want the validation for the last sequence, invert everything to generate metadata
            # Right now evaluating every sequence
            logger().info("Start validation:")
            variance_over_all_test_sequences = []
            for loop_sequence in collection_of_previous_sequences:

                next_validation_dataset, prev_dataset = self._dataset_loader.get_tf_dataset(sequence=loop_sequence,
                                                                                 mode="validation", repeat=False,
                                                                                 shuffle=False)

                if loop_sequence != current_sequence:

                    # update the validation set to match the current amount of classes
                    autotune = tf.data.experimental.AUTOTUNE
                    added_classes_since_loop = total_amount_of_classes - amount_of_classes_per_seq[loop_sequence]

                    prev_dataset = prev_dataset.map(lambda color_img, label, path: (color_img,
                                      expand_label(label, added_classes_since_loop), path),
                                      num_parallel_calls=autotune)
                    ready_for_intermediate = prev_dataset.map(lambda color_img, label, path: (color_img, label, label, path),
                                      num_parallel_calls=autotune)
                    intermediate_val_set = self._dataset_loader.finalize_dataset(ready_for_intermediate, repeat=False, shuffle=False)
                    self._model.change_to_dataset(intermediate_val_set)
                    next_validation_dataset, _ = self.add_reconstruction_labels_to_tf_dataset(prev_dataset, shuffle=False, repeat=False)

                if self._config.record_tensorboard:
                    self._model.create_new_file_writer(loop_sequence, "validation")
                accuracy, variances_in_last_layer = self._model.run_validation_for_dataset(next_validation_dataset)
                list_of_validation_accs.append(accuracy)
                variance_over_all_test_sequences.append(np.mean(variances_in_last_layer))
                #creating accuracy yaml
                if current_sequence in self._acc_dict:
                    self._acc_dict[current_sequence][loop_sequence] = float(accuracy)
                else:
                    self._acc_dict[current_sequence] = {loop_sequence: float(accuracy)}
                logger().info("\tDone with validation sequence: {}, has acc: {:.5f}".format(loop_sequence, accuracy))
            list_of_variances_per_sequence.append(np.mean(variance_over_all_test_sequences))


        # generating metadata.txt: with all the data used for the CLScore
        elapsed_time = (time.time() - start_time) / 60
        logger().info("Done with training in: {}m".format(elapsed_time))
        logger().info("Start with testing")

        #create yaml with all accuracys
        new_keys = {}
        for key in self._acc_dict.keys():
            new_keys[key] = np.mean([element for element in self._acc_dict[key].values()])
        for key, value in new_keys.items():
            self._acc_dict[key + "_average"] = value
            print("Average for {}: {}".format(key, value))
        with (self._config.tensorboard_path / "accuracys.yml").open('w') as f:
            f.write(yaml.dump(self._acc_dict))
        final_acc = np.mean([ele for _, ele in self._acc_dict[self.order_of_sequences[-1]].items()])
        logger().info("Final acc: {}".format(final_acc))

    def add_reconstruction_labels_to_tf_dataset(self, prev_dataset, shuffle, repeat=True):
        rec_labels = []
        while True:
            try:
                rec_label = self._sess.run(self._model._final_dense_layer, feed_dict=self._model.get_current_batch_dict(True))
                rec_labels.extend(rec_label)
            except tf.errors.OutOfRangeError:
                break
        rec_labels = np.array(rec_labels)
        varianz_per_class = np.var(rec_labels, axis=0)
        pred_dataset = tf.data.Dataset.from_tensor_slices(rec_labels)
        final_dataset = tf.data.Dataset.zip((prev_dataset, pred_dataset))

        autotune = tf.data.experimental.AUTOTUNE
        # only changes the braces part = (color_img, true_label, path)
        final_dataset = final_dataset.map(lambda part, rec_label: (part[0], part[1], rec_label, part[2]),
                              num_parallel_calls=autotune)
        final_dataset = self._dataset_loader.finalize_dataset(final_dataset, repeat=repeat, shuffle=shuffle)
        return final_dataset, varianz_per_class

