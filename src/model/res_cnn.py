# -*- coding: utf-8 -*-

import tensorflow as tf
import random

"""
@Author: xiezizhe 
@Date: 2018/11/15 5:15 PM
"""
LOOP_SIZE = 2
FILTER_DEPTH = 128
OUTPUT_SIZE = 512


class ResCNN:

    def __init__(self, max_length, embedding_size=128, trainable=True,
                 regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):
        self.positive_input = tf.placeholder(tf.float32, [None, max_length, embedding_size], name="positive_input")
        self.anchor_input = tf.placeholder(tf.float32, [None, max_length, embedding_size], name="anchor_input")
        self.negative_input = tf.placeholder(tf.float32, [None, max_length, embedding_size], name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")
        self.regularizer = regularizer
        self.trainable = trainable

        with tf.variable_scope("triplet"):
            self.anchor_output = self.network(self.anchor_input, reuse=False)
            self.positive_output = self.network(self.positive_input, reuse=True)
            self.negative_output = self.network(self.negative_input, reuse=True)

        with tf.name_scope("cosine_similarity"):
            self.pos_sim = self.cosine(self.anchor_output, self.positive_output)
            self.neg_sim = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("loss"):
            self.loss = self.cal_loss()
            if regularizer is not None:
                self.loss += tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu()
            tf.summary.scalar('accuracy', self.accuracy)

    def network(self, input_tensor, mid_channels=32, depth=1, reuse=False):
        output = tf.layers.conv2d(tf.expand_dims(input_tensor, -1), mid_channels, 3,
                                  padding='same', activation=tf.nn.relu)

        for layers in range(depth):
            with tf.variable_scope('block_{}'.format(layers)):
                identity = tf.layers.conv2d(output, mid_channels, 1, padding='same', name='identity_conv',
                                            reuse=reuse, use_bias=False)
                identity = tf.layers.batch_normalization(identity, training=self.training, reuse=reuse, name="bn1")
                output = tf.layers.conv2d(output, mid_channels // 4, [1, output.shape[2].value // 8], padding='same',
                                          name='conv_{}_0'.format(layers), use_bias=False, reuse=reuse)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training=self.training,
                                                                  reuse=reuse, name="bn2"))

                output = tf.layers.conv2d(output, mid_channels, 1, padding='same', name='conv_{}_2'.format(layers),
                                          use_bias=False, reuse=reuse)
                output = tf.layers.batch_normalization(output, training=self.training, reuse=reuse, name="bn3")
                output = tf.nn.relu(identity + output)

        with tf.variable_scope('blockY'):
            output = tf.layers.conv2d(output, 2, [1, output.shape[2].value // 4], padding='same',
                                      use_bias=False, reuse=reuse)

        flatten = tf.layers.flatten(output)
        if not reuse:
            print("flatten shape {}".format(flatten.shape))
        fcl = tf.layers.dense(flatten, OUTPUT_SIZE, name="fcl", reuse=reuse)
        return tf.layers.dropout(fcl, training=self.training, name="output")

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
                               tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss(self):
        logits = tf.nn.softmax(tf.stack([self.pos_sim, self.neg_sim], axis=1))
        right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        loss = tf.square(1 + right)
        return tf.reduce_mean(loss)

    def cal_accu(self):
        mean = tf.cast(tf.argmax(tf.stack([self.neg_sim, self.pos_sim], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)
