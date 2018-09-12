# -*- coding: utf-8 -*-
import tensorflow as tf


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
            self.positive_output = self.network(self.positive_input, reuse=None)
            self.anchor_output = self.network(self.anchor_input, reuse=True)
            self.negative_output = self.network(self.negative_input, reuse=True)

        with tf.name_scope("negative_cosine_distance"):
            self.d_pos = self.cosine(self.anchor_output, self.positive_output)

        with tf.name_scope("positive_cosine_distance"):
            self.d_neg = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("loss"):
            self.loss = self.cal_loss()

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu()

    def network(self, x, reuse=True):
        cnn1 = tf.layers.conv2d(tf.expand_dims(x, -1), filters=256, kernel_size=[3, 256],
                                padding="VALID", activation=tf.nn.tanh, name="cnn1", reuse=reuse)
        # bn1 = tf.layers.batch_normalization(cnn1, name="bn1", reuse=reuse, training=self.training)

        cnn2 = tf.layers.conv2d(cnn1, filters=512, kernel_size=[5, 1],
                                padding="VALID", activation=tf.nn.tanh, name="cnn2", reuse=reuse)
        # bn2 = tf.layers.batch_normalization(cnn2, name="bn2", reuse=reuse, training=self.training)

        flattern = tf.layers.flatten(cnn2, 'flatten_layer')

        fcl1 = tf.layers.dense(flattern, 256, name="fcl1", activation=tf.nn.tanh, reuse=reuse)
        fcl1 = tf.layers.dropout(fcl1, training=self.training)

        out = tf.layers.dense(fcl1, 128, name="output", activation=tf.nn.tanh, reuse=reuse)
        # print("{}->{}->{}->{}->{}->{}".format(cnn1.name, bn1.name, cnn2.name, bn2.name, fcl1.name, out.name))
        print("{}->{}->{}->{}".format(cnn1.name, cnn2.name, fcl1.name, out.name))
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
