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
Class for the RECALL model
"""

__author__ = "Markus Knauer, Maximilian Denninger, Rudolph Triebel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Markus Knauer"
__email__ = "markus.knauer@dlr.de"

import os
import numpy as np
import tensorflow.compat.v1 as tf

# change tensorflow logger to only let through above info
tf.get_logger().setLevel('INFO')

from tensorflow.python.keras import layers
from recall.utility.config import Configuration
from recall.data_import.dataset_loader import DataSetLoader


class RecallModel(object):

    def __init__(self, dataset_loader: DataSetLoader, config: Configuration):
        """
        Recall model introduced in the RECALL paper.

        :param dataset_loader: the used dataset loader
        :param config: the used config
        """
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        self._dataset_loader = dataset_loader
        self._config = config

        self._res_net_features = self._dataset_loader.get_input()
        self._is_training_placeholders = {}
        self._variable_collection = {}

        self._true_labels, self._rec_labels, _ = dataset_loader.get_input_values()
        self.iterator = dataset_loader.get_iterator()
        self._final_softmax = None
        self._amount_of_classes = 0
        self._amount_of_classes_list = [0]
        self._variables_to_train = None
        self._final_dense_layer = None
        self._list_of_dense_layers = []
        self._list_of_accs = []
        self._file_writer = None
        self._list_of_summaries = []
        self._list_of_keep_summaries = []
        self._expanding_acc_op = None
        self._seq_counter = 0
        self._adam = tf.train.AdamOptimizer(1e-5)
        self._list_of_names = []
        self._variance_per_class = []
        # this collection contains all used layers per sequence
        self._collection_of_layers = {}
        self._current_session = None
        self._used_sessions = []

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self._sess = tf.Session(config=tf_config)
        self._sess.run(tf.global_variables_initializer())
        self._collection_of_already_init_vars = set(tf.all_variables())

    def _update_current_session(self):
        """
        Updates the current session, which also adds an empty list to the collection of layers, to save all
        relevant current layers, which could be turned of during training.
        """
        if self._current_session is None:
            self._current_session = "base"
        elif self._current_session == "base":
            self._current_session = "seq1"
        else:
            self._current_session = "seq" + str(int(self._current_session[len("seq"):]) + 1)
        self._collection_of_layers[self._current_session] = []
        self._used_sessions.append(self._current_session)

    def create_dense_layer(self, mode: str):
        """
        Create a dense layer for head
        """
        with tf.name_scope("head"):
            # this value is fixed after extensive testing
            x = self._res_net_features

            x = layers.Dense(2048, kernel_initializer="glorot_uniform", name='fc_2048')(x)
            self._collection_of_layers[self._current_session].append(x)

            # apply siren activation
            x = self._config.siren_omega * x
            x = tf.math.sin(x)

            return x

    def change_to_dataset(self, dataset: tf.data.Dataset):
        """
        Switch to current tf.data.dataset
        """
        train_init_op = self.iterator.make_initializer(dataset)
        self._sess.run(train_init_op)
        return train_init_op

    def run_validation_for_dataset(self, dataset: tf.data.Dataset):
        self.change_to_dataset(dataset)
        # reset all tracker counter
        self._sess.run(tf.local_variables_initializer())
        list_of_operations = self.get_list_of_ops_for("validation")
        if self._config.using_reconstruction_loss:
            clipped_values = tf.clip_by_value(self._final_dense_layer, 0, 1)
            list_of_operations.append(tf.math.reduce_variance(clipped_values))
        else:
            list_of_operations.append(tf.math.reduce_variance(self._final_dense_layer))
        counter, accuracy = 0, 0
        labels, predictions, variances_in_last_layer = [], [], []
        while True:
            try:
                result = self._sess.run(list_of_operations, feed_dict=self.get_current_batch_dict(True))
                variances_in_last_layer.append(result[-1])
                accuracy = result[-3]
                predictions.extend(result[-4])
                labels.extend(result[-5])
                if self._file_writer is not None:
                    for summary in result[-2]:
                        self._file_writer.add_summary(summary, global_step=counter)
                    counter += 1
            except tf.errors.OutOfRangeError:
                break

        return accuracy, variances_in_last_layer

    def add_new_classes(self, number_of_new_classes):
        """
        Adds new outputs to the last layer
        :param number_of_new_classes: number of new classes
        """

        self._list_of_summaries = []
        assign_op_list = []
        old_amount_of_classes = self._amount_of_classes
        self._amount_of_classes_list.append(self._amount_of_classes)
        self.old_amount_of_cl = old_amount_of_classes
        self._amount_of_classes += number_of_new_classes

        self._update_current_session()

        if self._final_dense_layer is None and self._final_softmax is None:
            scope_name = "base"
            with tf.name_scope(scope_name):
                # There were no classes before
                new_name = "dense_layer_before_softmax"
                self._last_layer_of_backbone = self.create_dense_layer("base")
                kernel_initializer = 'glorot_uniform'
                dense_layer = layers.Dense(self._amount_of_classes, kernel_initializer=kernel_initializer,
                                           name=new_name)(self._last_layer_of_backbone)
                self._collection_of_layers[self._current_session].append(dense_layer)
                self._final_dense_layer = dense_layer
                self._list_of_names.append(scope_name + "/" + new_name)
                self._list_of_dense_layers.append(self._final_dense_layer)

                self._final_softmax = layers.Softmax(name="Softmax")(self._final_dense_layer)

                # create current accuracy

                self._labels = tf.argmax(self._true_labels, 1)
                self._predictions = tf.argmax(self._final_softmax, 1)

                _, self._total_acc_op = tf.metrics.accuracy(labels=self._labels,
                                                            predictions=self._predictions)
                if self._config.record_tensorboard:
                    acc_sum = tf.summary.scalar("Total_Acc_accumluate", self._total_acc_op)
                    self._list_of_summaries.append(acc_sum)

                # to avoid running the same operation twice
                _, total_acc_classes_op = tf.metrics.accuracy(
                    labels=tf.argmax(self._true_labels[:, :self._amount_of_classes], 1),
                    predictions=tf.argmax(self._final_softmax, 1))
                self._list_of_accs.append(total_acc_classes_op)
                if self._config.record_tensorboard:
                    variance_per_seq = tf.summary.scalar("Var_{}".format(self._seq_counter), tf.math.reduce_variance(
                        self._final_dense_layer[:, old_amount_of_classes: self._amount_of_classes]))
                    self._list_of_keep_summaries.append(variance_per_seq)

                # add loss

                if self._config.using_reconstruction_loss:
                    clipped_values = tf.clip_by_value(self._final_dense_layer, 0, 1)
                    train_loss = tf.reduce_mean(tf.math.square(self._true_labels - clipped_values))
                else:
                    train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._true_labels,
                                                                            logits=self._final_dense_layer)

                self.loss = train_loss

                if self._config.record_tensorboard:
                    train_loss_sum = tf.summary.scalar("Trainings_loss", tf.math.reduce_mean(train_loss))
                    self._list_of_summaries.append(train_loss_sum)

                self._variables_to_train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

                self._variable_collection["base"] = self._variables_to_train

                # self._train_op = self._adam.minimize(train_loss, var_list=self._variables_to_train)
                self._train_op = tf.train.AdamOptimizer().minimize(train_loss, var_list=self._variables_to_train)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self._train_op = tf.group([self._train_op, update_ops])
        else:
            self._seq_counter += 1
            scope_name = "seq_{}".format(self._seq_counter)
            with tf.name_scope(scope_name):
                new_name = "dense_layer_before_softmax_new_seq_{}".format(self._seq_counter)
                current_last_layer = self.create_dense_layer("seq{}".format(self._seq_counter))
                new_dense_layer = tf.keras.layers.Dense(number_of_new_classes, kernel_initializer="glorot_uniform",
                                                        name=new_name)(current_last_layer)
                self._collection_of_layers[self._current_session].append(new_dense_layer)
                new_layer = new_dense_layer
                self._list_of_names.append(scope_name + "/" + new_name)

                self._list_of_dense_layers.append(new_layer)
                self._final_dense_layer = layers.concatenate(self._list_of_dense_layers)

                if self._config.record_tensorboard:
                    variance_per_seq = tf.summary.scalar("Var_{}".format(self._seq_counter), tf.math.reduce_variance(
                        self._final_dense_layer[:, old_amount_of_classes: self._amount_of_classes]))
                    self._list_of_keep_summaries.append(variance_per_seq)

                # is only used in metrics accuracy
                self._final_softmax = layers.Softmax(name="Softmax_concatenate")(self._final_dense_layer)

                # create current accuracy
                self._labels = tf.argmax(self._true_labels, 1)
                self._predictions = tf.argmax(self._final_softmax, 1)
                _, self._total_acc_op = tf.metrics.accuracy(labels=self._labels,
                                                            predictions=self._predictions)

                if self._config.record_tensorboard:
                    acc_sum = tf.summary.scalar("Total_Acc_accumluate", self._total_acc_op)
                    self._list_of_summaries.append(acc_sum)
                _, new_acc_classes_op = tf.metrics.accuracy(
                    labels=tf.argmax(self._true_labels[:, self.old_amount_of_cl:self._amount_of_classes], 1,
                                     name="true_argmax_{}_{}".format(self.old_amount_of_cl, self._amount_of_classes)),
                    predictions=tf.argmax(self._final_softmax[:, self.old_amount_of_cl:self._amount_of_classes], 1),
                    name="pred_argmax_{}_{}".format(self.old_amount_of_cl, self._amount_of_classes))
                if self._config.using_reconstruction_loss:
                    self._final_dense_layer = tf.clip_by_value(self._final_dense_layer, 0, 1)
                self._list_of_accs.append(new_acc_classes_op)

                ## Our custom loss function for new classes
                print("old_amount_of_cl = {}".format(self.old_amount_of_cl))
                self._variance_per_class = tf.Variable(initial_value=np.zeros(self.old_amount_of_cl), dtype=tf.float32, name="varianz_per_class")
                org_reconstruction_layer_result = self._final_dense_layer[:, :self.old_amount_of_cl]
                # calculate the difference
                difference_old_loss = org_reconstruction_layer_result - self._rec_labels[:, :self.old_amount_of_cl]

                if self._config.record_tensorboard:
                    org_sum = tf.summary.histogram("Org_rec_{}".format(self._seq_counter),
                                                   org_reconstruction_layer_result)
                    self._list_of_keep_summaries.append(org_sum)
                    org_sum_1 = tf.summary.histogram("Rec_label_{}".format(self._seq_counter),
                                                     self._rec_labels[:, :self.old_amount_of_cl])
                    self._list_of_keep_summaries.append(org_sum_1)
                    diff = tf.summary.histogram("Diff_rec{}".format(self._seq_counter), difference_old_loss)
                    self._list_of_keep_summaries.append(diff)
                divide_org = org_reconstruction_layer_result
                if self._config.do_divide_with_var:
                    difference_old_loss = tf.math.divide(difference_old_loss, self._variance_per_class)

                    divide_org = tf.math.divide(org_reconstruction_layer_result, self._variance_per_class)
                for index in range(1, len(self._amount_of_classes_list) - 1):
                    old_classes, current_classes = self._amount_of_classes_list[index - 1], \
                                                   self._amount_of_classes_list[index]
                    if self._config.record_tensorboard:
                        variance_per_seq = tf.summary.scalar("Var_rec_{}_{}".format(self._seq_counter, index - 1),
                                                             tf.math.reduce_variance(org_reconstruction_layer_result[:,
                                                                                     old_classes: current_classes]))
                        self._list_of_keep_summaries.append(variance_per_seq)
                        variance_per_seq = tf.summary.scalar("Var_diff_{}_{}".format(self._seq_counter, index - 1),
                                                             tf.math.reduce_variance(
                                                                 divide_org[:, old_classes: current_classes]))
                        self._list_of_keep_summaries.append(variance_per_seq)
                if self._config.record_tensorboard:
                    variance_per_seq = tf.summary.scalar("Var_diff_{}_{}".format(self._seq_counter, self._seq_counter),
                                                         tf.math.reduce_variance(
                                                             self._final_dense_layer[:, self.old_amount_of_cl:]))
                    self._list_of_keep_summaries.append(variance_per_seq)

                if self._config.using_reconstruction_loss:
                    reconstruction_loss_on_old_weights = tf.reduce_mean(tf.math.square(difference_old_loss))
                    self.loss = reconstruction_loss_on_old_weights
                    reconstruction_loss_on_new_weights = tf.reduce_mean(
                        tf.math.square(self._true_labels[:, self.old_amount_of_cl:] -
                                       self._final_dense_layer[:, self.old_amount_of_cl:]))
                    reconstruction_loss_combined = tf.reduce_mean(
                        tf.math.square(self._true_labels - self._final_dense_layer))
                    if self._config.use_combined_loss:
                        train_loss = reconstruction_loss_on_old_weights + reconstruction_loss_on_new_weights + reconstruction_loss_combined
                    else:
                        train_loss = reconstruction_loss_on_old_weights + reconstruction_loss_on_new_weights
                    if self._config.record_tensorboard:
                        train_loss_sum_1 = tf.summary.scalar("Trainings_loss", tf.math.reduce_mean(train_loss))
                        train_loss_sum_2 = tf.summary.scalar("Loss_on_old",
                                                             tf.math.reduce_mean(reconstruction_loss_on_old_weights))
                        train_loss_sum_3 = tf.summary.scalar("Loss_on_old_scaled",
                                                             tf.math.reduce_mean(reconstruction_loss_on_old_weights))
                        train_loss_sum_4 = tf.summary.scalar("Loss_on_new",
                                                             tf.math.reduce_mean(reconstruction_loss_on_new_weights))
                        train_loss_sum_5 = tf.summary.scalar("Loss_combined",
                                                             tf.math.reduce_mean(reconstruction_loss_combined))
                        self._list_of_summaries.extend([train_loss_sum_1, train_loss_sum_2, train_loss_sum_3,
                                                        train_loss_sum_4, train_loss_sum_5])

                else:
                    loss_on_old_weights = tf.math.reduce_mean(tf.math.square(difference_old_loss))
                    self.loss = loss_on_old_weights
                    loss_on_new_weights = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=self._true_labels[:, self.old_amount_of_cl:],
                        logits=self._final_dense_layer[:, self.old_amount_of_cl:])
                    loss_combined = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._true_labels,
                                                                               logits=self._final_dense_layer)

                    if self._config.use_combined_loss:
                        train_loss = loss_on_old_weights + loss_on_new_weights + loss_combined
                    else:
                        train_loss = loss_on_old_weights + loss_on_new_weights

                    if self._config.record_tensorboard:
                        train_loss_sum_1 = tf.summary.scalar("Trainings_loss", tf.math.reduce_mean(train_loss))
                        train_loss_sum_2 = tf.summary.scalar("Loss_on_old", tf.math.reduce_mean(loss_on_old_weights))
                        train_loss_sum_3 = tf.summary.scalar("Loss_on_old_scaled",
                                                             tf.math.reduce_mean(loss_on_old_weights))
                        train_loss_sum_4 = tf.summary.scalar("Loss_on_new", tf.math.reduce_mean(loss_on_new_weights))
                        train_loss_sum_5 = tf.summary.scalar("Loss_combined", tf.math.reduce_mean(loss_combined))
                        self._list_of_summaries.extend([train_loss_sum_1, train_loss_sum_2, train_loss_sum_3,
                                                        train_loss_sum_4, train_loss_sum_5])

                # use all variables for training
                self._variables_to_train = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

                new_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                for key, variables in self._variable_collection.items():
                    new_variables = list(set(new_variables) - set(variables))
                self._variable_collection["seq{}".format(self._seq_counter)] = new_variables

                adam_opt = tf.train.AdamOptimizer(self._config.seq_learning_rate)
                self._train_op = adam_opt.minimize(train_loss, var_list=self._variables_to_train)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self._train_op = tf.group([self._train_op, update_ops])

                ## loading weights from previous heads
                if self._config.initialize_heads:
                    last_head_variable_list = []

                    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        if self._seq_counter == 1 or self._config.take_weights_from_base_head:
                            prev = "base"
                        else:
                            prev = "seq_{}".format(self._seq_counter - 1)
                        if variable.name.startswith(prev + "/head"):
                            last_head_variable_list.append(variable)

                    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                        if variable.name.startswith("seq_{}".format(self._seq_counter)):
                            for last_head_var in last_head_variable_list:
                                new_name = variable.name[variable.name.find("/") + 1:]
                                if new_name in last_head_var.name:
                                    if 'Adam' not in new_name and 'gamma' not in new_name and 'beta' not in new_name and 'batchnorm' not in new_name:
                                        assign_op_list.append(variable.assign(last_head_var))

        # get all variables, which have not been inited yet
        self._sess.run(tf.initialize_variables(set(tf.all_variables()) - self._collection_of_already_init_vars))
        # update all variables, which have been now inited
        self._collection_of_already_init_vars = set(tf.all_variables())
        if len(assign_op_list) > 0:
            self._sess.run(assign_op_list)

    def perform_train_step(self, list_of_operations, counter, trainings_dict):
        if hasattr(self._config, "freeze_old_weights") and self._config.freeze_old_weights:
            for session_name in self._used_sessions[:-1]:
                for layer in self._collection_of_layers[session_name]:
                    layer.trainable = False
            for layer in self._collection_of_layers[self._used_sessions[-1]]:
                layer.trainable = True

        result = self._sess.run(list_of_operations, feed_dict=trainings_dict)
        summaries = result[-1]
        if self._config.record_tensorboard:
            for summary in summaries:
                self._file_writer.add_summary(summary, global_step=counter)
        return result[:len(self._list_of_accs) + 1]

    def get_current_batch_dict(self, testing):
        feed_dict = {}
        if testing:
            for key, value in self._is_training_placeholders.items():
                feed_dict[value] = False
            return feed_dict
        find_latest_key = "base"
        highest_nr = -1
        for key in self._is_training_placeholders.keys():
            if "seq" in key and not "core" in key:
                nr = int(key[len("seq"):])
                if nr > highest_nr:
                    highest_nr = nr
                    find_latest_key = key
        for key, value in self._is_training_placeholders.items():
            if find_latest_key in key:
                feed_dict[value] = True
            else:
                feed_dict[value] = False
        # print(feed_dict)
        return feed_dict

    def get_list_of_ops_for(self, mode):
        list_of_operations = []
        if mode == "train":
            list_of_operations.extend(self._list_of_accs)
            list_of_operations.append(self._total_acc_op)
            list_of_operations.append(self._train_op)
            temp_sum = list(self._list_of_summaries)
            temp_sum.extend(self._list_of_keep_summaries)
            list_of_operations.append(temp_sum)
        elif mode == "validation":
            list_of_operations.extend(self._list_of_accs)
            list_of_operations.append(self._labels)
            list_of_operations.append(self._predictions)
            list_of_operations.append(self._total_acc_op)
            temp_sum = list(self._list_of_summaries)
            temp_sum.extend(self._list_of_keep_summaries)
            list_of_operations.append(temp_sum)
        else:
            raise Exception("Unknown mode: {}".format(mode))
        return list_of_operations

    def create_new_file_writer(self, sequence, mode):
        """
        Creates a tensorboard file to write in the tf summaries
        :param sequence: Number of training sequences
        :param mode: Training or Validation
        """
        if self._file_writer is not None:
            self._file_writer.close()
        log_dir = (self._config.tensorboard_path / sequence / mode).absolute()
        self._file_writer = tf.summary.FileWriter(log_dir, session=self._sess, flush_secs=10)  # graph=self._sess.graph)
