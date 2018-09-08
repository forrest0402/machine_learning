# -*- coding: utf-8 -*-
import tensorflow as tf

import src.loss.triplet_loss as triplet_loss


class TripletNetwork:
    """
    Triplet Network V3
    Add batch normalization
    """

    def __init__(self, length, embedding_size, train=False):
        self.train = train
        self.positive_input = tf.placeholder(tf.float32, [None, length, embedding_size])
        self.anchor_input = tf.placeholder(tf.float32, [None, length, embedding_size])
        self.negative_input = tf.placeholder(tf.float32, [None, length, embedding_size])

        with tf.variable_scope("triplet") as scope:
            self.positive_output = self.network(self.positive_input)
            scope.reuse_variables()
            self.anchor_output = self.network(self.anchor_input)
            self.negative_output = self.network(self.negative_input)

        self.d_pos = self.cosine(self.anchor_output, self.positive_output)
        self.d_neg = self.cosine(self.anchor_output, self.negative_output)
        self.loss = self.cal_loss()
        self.accuracy = self.cal_accu()

    def network(self, x):
        cnn1 = self.cnn_layer(tf.expand_dims(x, -1), "cnn1", [3, 256, 1, 512])
        cnn2 = self.cnn_layer(cnn1, "cnn2", [5, 1, 512, 1024])

        flatten_layer = tf.contrib.layers.flatten(cnn2, 'flatten_layer')
        weights = tf.get_variable(shape=[flatten_layer.shape[-1], 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  name='fc_weights')
        biases = tf.get_variable(shape=[128], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0), name='fc_biases')
        logit_output = tf.nn.tanh(tf.nn.bias_add(tf.matmul(flatten_layer, weights), biases,
                                                 name='logit_output'))

        out = tf.layers.dropout(logit_output, rate=0.5, training=self.train)
        return out

    def fc_layer(self, tensor, n_weight, name):
        n_prev_weight = tensor.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight],
                            initializer=initer)
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(tensor, W), b)
        return fc

    def cnn_layer(self, tensor, name, shape):
        conv_weights = tf.get_variable(name + "weight", shape,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_biases = tf.get_variable(name + "bias", [shape[3]],
                                      initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(tensor, conv_weights, strides=[1, 1, 1, 1], padding="VALID")
        tanh = tf.nn.tanh(tf.nn.bias_add(conv, conv_biases))
        bn = tf.layers.batch_normalization(tanh, training=self.train, reuse=tf.AUTO_REUSE,
                                           name=name)
        pool = tf.nn.max_pool(bn, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        return pool

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
                               tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss(self):
        margin = 0.3
        loss = tf.maximum(0.0, margin + self.d_pos - self.d_neg)
        return tf.reduce_mean(loss)

    def cal_accu(self):
        mean = tf.cast(tf.argmin(tf.stack([self.d_neg, self.d_pos], axis=1), axis=1),
                       dtype=tf.float32)
        return tf.reduce_mean(mean)