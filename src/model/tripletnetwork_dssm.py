# -*- coding: utf-8 -*-

import tensorflow as tf

FILTER_SIZES = [2, 3, 5, 7]
FILTER_DEPTH = 128


class TripletNetwork:
    """
    Triplet Network V4
    """

    # inputs should be int
    def __init__(self, length, embedding_size):
        self.positive_input = tf.placeholder(tf.float32, [None, length, embedding_size], name="positive_input")
        self.anchor_input = tf.placeholder(tf.float32, [None, length, embedding_size], name="anchor_input")
        self.negative_input = tf.placeholder(tf.float32, [None, length, embedding_size], name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")

        with tf.variable_scope("triplet"):
            self.positive_output, self.regularization = self.network(self.positive_input, embedding_size, reuse=None)
            self.anchor_output, _ = self.network(self.anchor_input, embedding_size, reuse=True)
            self.negative_output, _ = self.network(self.negative_input, embedding_size, reuse=True)

        with tf.name_scope("positive_cosine_distance"):
            self.positive_sim = self.cosine(self.anchor_output, self.positive_output)

        with tf.name_scope("negative_cosine_distance"):
            self.negative_sim = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("loss"):
            self.loss = self.cal_loss() + self.regularization * 1e-3

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
                                                   training=self.training, epsilon=1e-5)
                filter_list.append(bn)

                if not reuse:
                    print("cnn: {} pool: {}".format(cnn.shape, pool.shape))

        flatten = tf.layers.flatten(tf.concat(filter_list, axis=3), 'flatten_layer')
        print("flatten shape: {}".format(flatten.shape))
        output_fcl = tf.layers.dense(flatten, FILTER_DEPTH, name="fcl1", activation=tf.nn.tanh,  reuse=reuse)
        output_bn = tf.layers.batch_normalization(output_fcl, name="output_bn", reuse=reuse,
                                                  training=self.training, epsilon=1e-5)
        out = tf.layers.dropout(output_bn, training=self.training, name="output_dropout")

        return out, tf.nn.l2_loss(output_fcl)

    def l1norm(self, vec1, vec2):
        return tf.reduce_sum(tf.abs(tf.subtract(vec1, vec2)), axis=1)

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
                               tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss_l1(self):
        margin = 0.2
        pos_dist = self.l1norm(self.anchor_output, self.positive_output)
        neg_dist = self.l1norm(self.anchor_output, self.negative_output)
        loss = tf.maximum(0.0, margin + pos_dist - neg_dist)
        return tf.reduce_mean(loss)

    def cal_loss(self):
        margin = 0.2
        loss = tf.maximum(0.0, margin - self.positive_sim + self.negative_sim)
        return tf.reduce_mean(loss)

    def cal_accu(self):
        mean = tf.cast(tf.argmax(tf.stack([self.negative_sim, self.positive_sim], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)
