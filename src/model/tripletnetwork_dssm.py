# -*- coding: utf-8 -*-

import random

import tensorflow as tf

FILTER_SIZES = [2, 3, 5, 7]
FILTER_DEPTH = 128


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
            self.positive_output = self.network(self.embedded_positive, "positive", reuse=False)
            self.anchor_output = self.network(self.embedded_anchor, "anchor", reuse=True)
            self.negative_output = self.network(self.embedded_negative, "negative", reuse=True)

        with tf.name_scope("positive_cosine_distance"):
            self.positive_sim = self.cosine(self.anchor_output, self.positive_output)

        with tf.name_scope("negative_cosine_distance"):
            self.negative_sim = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("l1_distance"):
            self.post_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.positive_output))
            self.neg_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.negative_output))

        with tf.name_scope("loss"):
            self.loss = self.cal_loss_11()
            if regularizer is not None:
                self.loss += tf.losses.get_regularization_loss()  # tf.add_n(tf.get_collection("losses"))

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu_l1()

    def network(self, x, name, reuse=True):
        filter_list = []
        for i, filter_size in enumerate(FILTER_SIZES):
            with tf.variable_scope("layer{}".format(i)):
                cnn = tf.layers.conv2d(x, filters=FILTER_DEPTH, kernel_size=[filter_size, x.shape[2].value],
                                       activation=tf.nn.tanh, reuse=reuse, use_bias=False, name="cnn")

                pool = tf.layers.max_pooling2d(cnn, [cnn.shape[1].value, cnn.shape[2].value], [1, 1],
                                               name="pool-{}".format(name))
                bn = tf.layers.batch_normalization(pool, name="bn", reuse=reuse, training=self.training)
                filter_list.append(bn)

                if not reuse:
                    print("cnn: {} pool: {}".format(cnn.shape, pool.shape))

        flatten = tf.layers.flatten(tf.concat(filter_list, axis=3), 'flatten_layer')
        print("flatten shape: {}".format(flatten.shape))

        if self.regularizer is not None:
            output_fcl = tf.layers.dense(flatten, 128, name="fcl1", activation=tf.nn.tanh, use_bias=False,
                                         reuse=reuse, kernel_regularizer=self.regularizer)
        else:
            output_fcl = tf.layers.dense(flatten, 128, name="fcl1", activation=tf.nn.tanh, use_bias=False, reuse=reuse)
        output_bn = tf.layers.batch_normalization(output_fcl, name="output_bn", reuse=reuse, training=self.training)
        out = tf.layers.dropout(output_bn, training=self.training, name="output_dropout")

        # if not reuse and self.regularizer is not None:
        #     tf.add_to_collection('losses', self.regularizer(output_fcl))

        return out

    def l1norm(self, vec1, vec2):
        return tf.reduce_sum(tf.abs(tf.subtract(vec1, vec2)), axis=1)

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
                               tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss_l1(self):
        logits = tf.nn.softmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1))
        right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        if random.random() >= 0.5:
            loss = tf.square(tf.maximum(tf.subtract(1.0, right), 0.0))
        else:
            loss = tf.square(tf.subtract(1.0, right))
        return tf.reduce_mean(loss)

    def cal_loss(self):
        logits = tf.nn.softmax(tf.stack([self.negative_sim, self.positive_sim], axis=1))
        right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        loss = tf.square(tf.maximum(tf.subtract(1.0, right), 0.0)) + tf.square(tf.subtract(1.0, right))
        return tf.reduce_mean(loss)

    def cal_accu_l1(self):
        mean = tf.cast(tf.argmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)

    def cal_accu(self):
        mean = tf.cast(tf.argmax(tf.stack([self.negative_sim, self.positive_sim], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)
