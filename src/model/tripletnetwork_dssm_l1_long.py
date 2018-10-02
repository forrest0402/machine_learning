# -*- coding: utf-8 -*-

import random

import tensorflow as tf

FILTER_SIZES = [2, 3, 5, 7]
FILTER_DEPTH = 100
GLOBAL_VARIABLE = dict()


class TripletNetwork:
    """
    Triplet Network
    """

    # inputs should be int
    def __init__(self, max_length, word_vocab, regularizer=None):
        self.positive_input = tf.placeholder(tf.int32, [None, max_length], name="positive_input")
        self.anchor_input = tf.placeholder(tf.int32, [None, max_length], name="anchor_input")
        self.negative_input = tf.placeholder(tf.int32, [None, max_length], name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")
        self.regularizer = regularizer

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.get_variable("word_embedding", trainable=False,
                                initializer=tf.constant(word_vocab), dtype=tf.float32)

            self.embedded_positive = tf.expand_dims(tf.nn.embedding_lookup(W, self.positive_input), -1)
            self.embedded_anchor = tf.expand_dims(tf.nn.embedding_lookup(W, self.anchor_input), -1)
            self.embedded_negative = tf.expand_dims(tf.nn.embedding_lookup(W, self.negative_input), -1)

        with tf.variable_scope("triplet"):
            filter_list_anchor = []
            filter_list_pos = []
            filter_list_neg = []
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1)
            for i, filter_size in enumerate(FILTER_SIZES):
                with tf.variable_scope("filter_{}".format(filter_size)):
                    # anchor
                    cnn = tf.layers.conv2d(self.embedded_anchor, filters=FILTER_DEPTH,
                                           kernel_size=[filter_size, self.embedded_anchor.shape[2].value],
                                           kernel_initializer=kernel_initializer,
                                           reuse=False, use_bias=False, name="cnn")
                    bn = tf.layers.batch_normalization(cnn, name="bn", reuse=False, training=False,
                                                       epsilon=1e-5)
                    # bn, mean, var, beta, offset = self.batch_normalization(cnn, FILTER_DEPTH)
                    pool = tf.layers.max_pooling2d(tf.nn.tanh(bn), [bn.shape[1].value, bn.shape[2].value],
                                                   [1, 1], name="pool-anchor")

                    filter_list_anchor.append(pool)
                    # positive
                    cnn = tf.layers.conv2d(self.embedded_positive, filters=FILTER_DEPTH,
                                           kernel_size=[filter_size, self.embedded_anchor.shape[2].value],
                                           reuse=True, use_bias=False, name="cnn")

                    # bn = self.batch_same(cnn, mean, var, beta, offset)
                    bn = tf.layers.batch_normalization(cnn, name="bn", reuse=True, training=False, epsilon=1e-5)
                    pool = tf.layers.max_pooling2d(tf.nn.tanh(bn), [bn.shape[1].value, bn.shape[2].value],
                                                   [1, 1], name="pool-positive")

                    filter_list_pos.append(pool)
                    # negative
                    cnn = tf.layers.conv2d(self.embedded_negative, filters=FILTER_DEPTH,
                                           kernel_size=[filter_size, self.embedded_anchor.shape[2].value],
                                           reuse=True, use_bias=False, name="cnn")
                    bn = tf.layers.batch_normalization(cnn, name="bn", reuse=True, training=False, epsilon=1e-5)
                    # bn = self.batch_same(cnn, mean, var, beta, offset)
                    pool = tf.layers.max_pooling2d(tf.nn.tanh(bn), [bn.shape[1].value, bn.shape[2].value],
                                                   [1, 1], name="pool-negative")

                    filter_list_neg.append(pool)

            flatten_anchor = tf.layers.flatten(tf.concat(filter_list_anchor, axis=3), 'flatten_anchor')
            flatten_pos = tf.layers.flatten(tf.concat(filter_list_pos, axis=3), 'flatten_pos')
            flatten_neg = tf.layers.flatten(tf.concat(filter_list_neg, axis=3), 'flatten_neg')

            # fcl
            fcl_anchor = tf.layers.dense(flatten_anchor, FILTER_DEPTH, name="fcl1", use_bias=True,
                                         reuse=False, kernel_regularizer=self.regularizer,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         bias_initializer=tf.constant_initializer(0.1))
            # output_bn, mean, var, beta, offset = self.batch_normalization(fcl_anchor, FILTER_DEPTH)
            output_bn = tf.layers.batch_normalization(fcl_anchor, name="output_bn", reuse=False, training=False,
                                                      epsilon=1e-5)
            self.anchor_output = tf.layers.dropout(tf.nn.tanh(output_bn), training=self.training,
                                                   name="output_dropout-anchor")

            fcl_pos = tf.layers.dense(flatten_pos, FILTER_DEPTH, name="fcl1", use_bias=True, reuse=True)
            # pos_bn = self.batch_same(fcl_pos, mean, var, beta, offset)
            pos_bn = tf.layers.batch_normalization(fcl_pos, name="output_bn", reuse=True, training=False,
                                                   epsilon=1e-5)
            self.positive_output = tf.layers.dropout(tf.nn.tanh(pos_bn), training=self.training,
                                                     name="output_dropout-pos")

            fcl_neg = tf.layers.dense(flatten_neg, FILTER_DEPTH, name="fcl1", use_bias=True, reuse=True)
            # neg_bn = self.batch_same(fcl_neg, mean, var, beta, offset)
            neg_bn = tf.layers.batch_normalization(fcl_neg, name="output_bn", reuse=True, training=False,
                                                   epsilon=1e-5)
            self.negative_output = tf.layers.dropout(tf.nn.tanh(neg_bn), training=self.training,
                                                     name="output_dropout-neg")

        with tf.name_scope("l1_distance"):
            self.post_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.positive_output))
            self.neg_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.negative_output))

        with tf.name_scope("loss"):
            self.loss = self.cal_loss_l1()
            if regularizer is not None:
                self.loss += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                # tf.losses.get_regularization_loss()  # tf.add_n(tf.get_collection("losses"))

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu_l1()

    def l1norm(self, vec1, vec2):
        return tf.reduce_sum(tf.abs(vec1 - vec2), axis=1)

    def cal_loss_l1(self):
        # batch_size = 128
        if random.random() >= 0.5:
            logits = tf.nn.softmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1))
            # labels = tf.constant([1] * batch_size, shape=[batch_size, 1], dtype=tf.float32)
            right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
            loss = tf.square(tf.maximum(tf.subtract(1.0, right), 0.0))
        else:
            logits = tf.nn.softmax(tf.stack([self.post_sim_l1, self.neg_sim_l1], axis=1))
            # labels = tf.constant([0] * batch_size, shape=[batch_size, 1], dtype=tf.float32)
            right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
            loss = tf.square(right)

        return tf.reduce_mean(loss)

    def cal_accu_l1(self):
        mean = tf.cast(tf.argmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)

    def batch_normalization(self, conv, num_filters, need_test=True):
        beta = tf.Variable(tf.ones(num_filters), name='beta')
        offset = tf.Variable(tf.zeros(num_filters), name='offset')
        bnepsilon = 1e-5

        axis = list(range(len(conv.shape) - 1))
        batch_mean, batch_var = tf.nn.moments(conv, axis)
        ema = tf.train.ExponentialMovingAverage(0.99)
        update_ema = ema.apply([batch_mean, batch_var])

        def mean_var_with_update(batch_mean, batch_var):
            with tf.control_dependencies([update_ema]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.equal(need_test, True),
                            lambda: mean_var_with_update(ema.average(batch_mean), ema.average(batch_var)),
                            lambda: (batch_mean, batch_var))

        Ybn = tf.nn.batch_normalization(conv, mean, var, offset, beta, bnepsilon)

        return Ybn, mean, var, beta, offset

    def batch_same(self, conv, mean, var, beta, offset):
        bnepsilon = 1e-5
        Ybn = tf.nn.batch_normalization(conv, mean, var, offset, beta, bnepsilon)
        return Ybn
