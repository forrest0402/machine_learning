# -*- coding: utf-8 -*-

import random

import tensorflow as tf

FILTER_SIZES = [2, 3, 5, 7]
FILTER_DEPTH = 100
GLOBAL_VARIABLE = dict()
FCL_LEN = 4


class TripletNetwork:
    """
    Triplet Network
    """

    def __init__(self, max_length, word_vocab, regularizer=None, trainable=True):
        self.positive_input = tf.placeholder(tf.int32, [None, max_length], name="positive_input")
        self.anchor_input = tf.placeholder(tf.int32, [None, max_length], name="anchor_input")
        self.negative_input = tf.placeholder(tf.int32, [None, max_length], name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")
        self.regularizer = regularizer
        self.trainable = trainable

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.get_variable("word_embedding", trainable=False,
                                initializer=tf.constant(word_vocab), dtype=tf.float32)

            self.embedded_positive = tf.expand_dims(tf.nn.embedding_lookup(W, self.positive_input), -1)
            self.embedded_anchor = tf.expand_dims(tf.nn.embedding_lookup(W, self.anchor_input), -1)
            self.embedded_negative = tf.expand_dims(tf.nn.embedding_lookup(W, self.negative_input), -1)

        with tf.variable_scope("triplet"):
            self.anchor_output = self.network(self.embedded_anchor, "anchor", False)
            self.positive_output = self.network(self.embedded_positive, "pos", True)
            self.negative_output = self.network(self.embedded_negative, "neg", True)

        with tf.name_scope("l1_sim"):
            self.post_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.positive_output))
            self.neg_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.negative_output))

        with tf.name_scope("loss"):
            self.loss = self.cal_loss_l1()
            if regularizer is not None:
                self.loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu_l1()

    def network(self, input, name, reuse=False):
        filter_list = []
        for i, filter_size in enumerate(FILTER_SIZES):
            with tf.variable_scope("filter_{}".format(filter_size)):
                cnn = tf.layers.conv2d(input, filters=FILTER_DEPTH, trainable=self.trainable,
                                       kernel_size=[filter_size, self.embedded_anchor.shape[2].value],
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       reuse=reuse, use_bias=False, name="cnn")
                bn = tf.layers.batch_normalization(cnn, name="bn", reuse=reuse, training=False,
                                                   trainable=self.trainable)
                pool = tf.layers.max_pooling2d(tf.nn.tanh(bn), [bn.shape[1].value, bn.shape[2].value],
                                               [1, 1], name="pool-{}".format(name))
                filter_list.append(pool)

        flatten = tf.layers.flatten(tf.concat(filter_list, axis=3), 'flatten-{}'.format(name))

        # fcl
        output = flatten
        for i in range(FCL_LEN):
            with tf.variable_scope("fcl_{}".format(i)):
                fcl = tf.layers.dense(output, FILTER_DEPTH, name="fcl", use_bias=True, reuse=reuse,
                                      activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.1))
                output = tf.layers.batch_normalization(fcl, name="bn", reuse=reuse, training=False)

        fcl = tf.layers.dense(output, FILTER_DEPTH, name="output_fcl", use_bias=True,
                              reuse=reuse, kernel_regularizer=self.regularizer,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.constant_initializer(0.1))
        output_bn = tf.layers.batch_normalization(fcl, name="output_bn", reuse=reuse, training=False, epsilon=1e-5)
        return tf.layers.dropout(tf.nn.tanh(output_bn), training=self.training, name="output-dropout-{}".format(name))

    def l1norm(self, vec1, vec2):
        return tf.reduce_sum(tf.abs(vec1 - vec2), axis=1)

    def cal_loss_l1(self):
        if random.random() >= 0.5:
            logits = tf.nn.softmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1))
            right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
            loss = tf.square(tf.maximum(tf.subtract(1.0, right), 0.0))
        else:
            logits = tf.nn.softmax(tf.stack([self.post_sim_l1, self.neg_sim_l1], axis=1))
            right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
            loss = tf.square(right)
        return tf.reduce_mean(loss)

    def cal_accu_l1(self):
        mean = tf.cast(tf.argmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)
