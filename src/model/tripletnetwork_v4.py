# -*- coding: utf-8 -*-
import tensorflow as tf

FILTER_SIZES = [2, 3, 3, 5, 7]
FILTER_DEPTH = 128


class TripletNetwork:
    """
    Triplet Network V4
    """

    # inputs should be int
    def __init__(self, length, embedding_size):
        self.positive_input = tf.placeholder(tf.float32, [None, length, embedding_size],
                                             name="positive_input")
        self.anchor_input = tf.placeholder(tf.float32, [None, length, embedding_size],
                                           name="anchor_input")
        self.negative_input = tf.placeholder(tf.float32, [None, length, embedding_size],
                                             name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")

        with tf.variable_scope("triplet"):
            self.positive_output = self.network(self.positive_input, embedding_size, reuse=None)
            self.anchor_output = self.network(self.anchor_input, embedding_size, reuse=True)
            self.negative_output = self.network(self.negative_input, embedding_size, reuse=True)

        with tf.name_scope("positive_cosine_distance"):
            self.d_pos = self.cosine(self.anchor_output, self.positive_output)

        with tf.name_scope("negative_cosine_distance"):
            self.d_neg = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("loss"):
            self.loss = self.cal_loss()

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu()

    def network(self, x, embedding_size, reuse=True):
        filter_list = []
        for i, filter_size in enumerate(FILTER_SIZES):
            with tf.variable_scope("filter{}".format(i)):
                cnn = tf.layers.conv2d(tf.expand_dims(x, -1), filters=FILTER_DEPTH,
                                       kernel_size=[filter_size, embedding_size],
                                       padding="VALID", activation=tf.nn.tanh, reuse=reuse,
                                       use_bias=False, name="cnn{}".format(i))
                pool = tf.layers.max_pooling2d(cnn, [cnn.shape[1], 1], [1, 1],
                                               name="pool{}".format(i))
                bn = tf.layers.batch_normalization(pool, name="bn{}".format(i), reuse=reuse,
                                                   training=self.training)
                filter_list.append(bn)

                if i == 0:
                    print("cnn: {} pool: {}".format(cnn.shape, pool.shape))

        flatten = tf.layers.flatten(tf.concat(filter_list, axis=3), 'flatten_layer')

        fcl1 = tf.layers.dense(flatten, 128, name="fcl1", activation=tf.nn.tanh, reuse=reuse)
        fcl1 = tf.layers.dropout(fcl1, training=self.training, name="dropout", rate=0.2)

        out = tf.layers.dense(fcl1, 128, name="output", reuse=reuse)
        return out

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
                               tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss(self):
        margin = 0.2
        loss = tf.maximum(0.0, margin + self.d_pos - self.d_neg)
        return tf.reduce_mean(loss)

    def cal_accu(self):
        mean = tf.cast(tf.argmin(tf.stack([self.d_neg, self.d_pos], axis=1), axis=1),
                       dtype=tf.float32)
        return tf.reduce_mean(mean)
