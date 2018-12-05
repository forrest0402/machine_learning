# -*- coding: utf-8 -*-

import tensorflow as tf

FILTER_SIZES = [2, 3, 5, 7]
FILTER_DEPTH = 128
OUTPUT_SIZE = 1024
GLOBAL_VARIABLE = dict()


class TripletNetwork:
    """
    Triplet Network
    """

    def __init__(self, max_length, embedding_dim=128, trainable=True,
                 regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):

        self.positive_input = tf.placeholder(tf.float32, [None, max_length, embedding_dim], name="positive_input")
        self.anchor_input = tf.placeholder(tf.float32, [None, max_length, embedding_dim], name="anchor_input")
        self.negative_input = tf.placeholder(tf.float32, [None, max_length, embedding_dim], name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")
        self.regularizer = regularizer
        self.trainable = trainable

        with tf.variable_scope("triplet"):
            self.anchor_output = self.network(tf.expand_dims(self.anchor_input, -1), "anchor", OUTPUT_SIZE, False)
            self.positive_output = self.network(tf.expand_dims(self.positive_input, -1), "pos", OUTPUT_SIZE, True)
            self.negative_output = self.network(tf.expand_dims(self.negative_input, -1), "neg", OUTPUT_SIZE, True)

        with tf.name_scope("cosine_similarity"):
            self.pos_sim = self.cosine(self.anchor_output, self.positive_output)
            self.neg_sim = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("loss"):
            self.loss = self.cal_loss1() + self.cal_loss2()
            if regularizer is not None:
                self.loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu()

    def attention(self, state, input, max_length):
        w, h = input.shape[1], input.shape[2]
        # state = tf.reshape(tf.concat([state] * max_length, axis=1), [-1, w, h])
        # attention = tf.nn.softmax(tf.reshape(self.cosine(state, input), shape=[-1, w * h]), axis=1)
        state = tf.reshape(tf.matmul(input, tf.expand_dims(state, -1)), [-1, max_length])
        attention = tf.nn.softmax(state, axis=1)
        return tf.multiply(input, tf.reshape(tf.concat([attention] * h, axis=1), [-1, w, h]))

    def network(self, input, name, output_nodes, reuse=False):
        filter_list = []
        for i, filter_size in enumerate(FILTER_SIZES):
            with tf.variable_scope("filter_{}".format(filter_size)):
                cnn = tf.layers.conv2d(input, filters=FILTER_DEPTH,
                                       kernel_size=[filter_size, self.anchor_input.shape[2].value],
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       reuse=reuse, use_bias=False, name="cnn", trainable=self.trainable)
                bn = tf.layers.batch_normalization(cnn, name="bn", reuse=reuse, training=False,
                                                   trainable=self.trainable)
                pool = tf.layers.max_pooling2d(tf.nn.tanh(bn), [bn.shape[1].value, bn.shape[2].value], [1, 1],
                                               name="pool{}".format(name))
                filter_list.append(pool)

        flatten = tf.layers.flatten(tf.concat(filter_list, axis=3), 'flatten{}'.format(name))

        # fcl
        fcl = tf.layers.dense(flatten, output_nodes, name="fcl1", use_bias=True, reuse=reuse,
                              kernel_regularizer=self.regularizer,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.constant_initializer(0.1))
        output_bn = tf.layers.batch_normalization(fcl, name="output_bn", reuse=reuse, training=False)
        output = tf.layers.dropout(tf.nn.tanh(output_bn), training=self.training, name="output-dropout{}".format(name))
        return output

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
                               tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss1(self):
        margin = 0.1
        loss = tf.maximum(0.0, margin + self.neg_sim - self.pos_sim)
        return tf.reduce_mean(loss)

    def cal_loss2(self):
        logits = tf.nn.softmax(tf.stack([self.neg_sim, self.pos_sim], axis=1))
        right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        loss = tf.square(tf.maximum(tf.subtract(1.0, right), 0.0))
        return tf.reduce_mean(loss)

    def cal_accu(self):
        mean = tf.cast(tf.argmax(tf.stack([self.neg_sim, self.pos_sim], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)
