# -*- coding: utf-8 -*-

import random

import tensorflow as tf

FILTER_SIZES = [2, 3, 5, 7]
FILTER_DEPTH = 128


class TripletNetwork:
    """
    Triplet Network V4
    """

    # inputs should be int
    def __init__(self, length, embedding_size, regularizer=None):
        self.positive_input = tf.placeholder(tf.float32, [None, length, embedding_size], name="positive_input")
        self.anchor_input = tf.placeholder(tf.float32, [None, length, embedding_size], name="anchor_input")
        self.negative_input = tf.placeholder(tf.float32, [None, length, embedding_size], name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")
        self.regularizer = regularizer

        with tf.variable_scope("triplet"):
            self.positive_output = self.network(self.positive_input, embedding_size, reuse=False)
            self.anchor_output = self.network(self.anchor_input, embedding_size, reuse=True)
            self.negative_output = self.network(self.negative_input, embedding_size, reuse=True)

        with tf.name_scope("positive_cosine_distance"):
            self.positive_sim = self.cosine(self.anchor_output, self.positive_output)

        with tf.name_scope("negative_cosine_distance"):
            self.negative_sim = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("l1_distance"):
            self.pos_dist = tf.negative(self.l1norm(self.anchor_output, self.positive_output))
            self.neg_dist = tf.negative(self.l1norm(self.anchor_output, self.negative_output))

        with tf.name_scope("loss"):
            self.loss = self.cal_loss()
            if regularizer is not None:
                self.loss += tf.add_n(tf.get_collection("losses"))

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu()

    def network(self, x, embedding_size, reuse=True):
        filter_list = []
        for i, filter_size in enumerate(FILTER_SIZES):
            with tf.variable_scope("layer{}".format(i)):
                cnn = tf.layers.conv2d(tf.expand_dims(x, -1), filters=FILTER_DEPTH,
                                       kernel_size=[filter_size, embedding_size],
                                       activation=tf.nn.tanh, reuse=reuse,
                                       use_bias=False, name="cnn")

                pool = tf.layers.max_pooling2d(cnn, [cnn.shape[1].value, cnn.shape[2].value],
                                               [1, 1], name="pool")
                bn = tf.layers.batch_normalization(pool, name="bn", reuse=reuse,
                                                   training=self.training)
                filter_list.append(bn)

                if not reuse:
                    print("cnn: {} pool: {}".format(cnn.shape, pool.shape))

        flatten = tf.layers.flatten(tf.concat(filter_list, axis=3), 'flatten_layer')
        print("flatten shape: {}".format(flatten.shape))
        output_fcl = tf.layers.dense(flatten, 128, name="fcl1", activation=tf.nn.tanh, reuse=reuse)
        output_bn = tf.layers.batch_normalization(output_fcl, name="output_bn", reuse=reuse,
                                                  training=self.training)
        out = tf.layers.dropout(output_bn, training=self.training, name="output_dropout")

        if not reuse and self.regularizer is not None:
            tf.add_to_collection('losses', self.regularizer(output_fcl))

        return out

    def l1norm(self, vec1, vec2):
        return tf.reduce_sum(tf.abs(tf.subtract(vec1, vec2)), axis=1)

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
                               tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss_l1(self):
        if random.random() >= 0.5:
            logits = tf.nn.softmax(tf.stack([self.neg_dist, self.pos_dist], axis=1))
            right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
            loss = tf.square(tf.maximum(tf.subtract(1.0, right), 0.0))
        else:
            logits = tf.nn.softmax(tf.stack([self.pos_dist, self.neg_dist], axis=1))
            right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
            loss = tf.square(right)
        return tf.reduce_mean(loss)

    def cal_loss(self):
        logits = tf.nn.softmax(tf.stack([self.negative_sim, self.positive_sim], axis=1))
        right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        loss = tf.square(tf.maximum(tf.subtract(1.0, right), 0.0)) + tf.square(tf.subtract(1.0, right))
        return tf.reduce_mean(loss)

    def cal_accu_l1(self):
        mean = tf.cast(tf.argmax(tf.stack([self.neg_dist, self.pos_dist], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)

    def cal_accu(self):
        mean = tf.cast(tf.argmax(tf.stack([self.negative_sim, self.positive_sim], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)
