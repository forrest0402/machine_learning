# -*- coding: utf-8 -*-
import tensorflow as tf

import src.loss.triplet_loss as triplet_loss


class TripletNetwork:

    def __init__(self, length, embedding_size):
        self.hidden_layer_num = length * 64

        self.positive_input = tf.placeholder(tf.float32, [None, length, embedding_size])
        self.anchor_input = tf.placeholder(tf.float32, [None, length, embedding_size])
        self.negative_input = tf.placeholder(tf.float32, [None, length, embedding_size])

        with tf.variable_scope("triplet") as scope:
            self.positive_output = self.network(self.positive_input)
            scope.reuse_variables()
            self.anchor_output = self.network(self.anchor_input)
            self.negative_output = self.network(self.negative_input)

        self.d_pos = self.cosine(self.anchor_output, self.positive_output)
        # tf.reduce_sum(tf.abs(self.anchor_output - self.positive_output), 1)
        self.d_neg = self.cosine(self.anchor_output, self.negative_output)
        # tf.reduce_sum(tf.abs(self.anchor_output - self.negative_output), 1)
        self.loss = self.cal_loss()
        self.accuracy = self.cal_accu()

    def network(self, x):
        node_num = x.get_shape()[1] * x.get_shape()[2]
        rs_x = tf.reshape(x, [-1, node_num])
        fc1 = self.fc_layer(rs_x, self.hidden_layer_num, "fc1")
        ac1 = tf.reshape(tf.nn.relu(fc1),
                         [-1, x.get_shape()[1].value, int(self.hidden_layer_num / x.get_shape()[1].value), 1])
        cnn1 = self.cnn_layer(ac1, "cnn1", [3, 64, 1, 512])
        cnn2 = self.cnn_layer(cnn1, "cnn2", [5, 1, 512, 1024])

        cnn2_shape = cnn2.get_shape().as_list()
        nodes = cnn2_shape[1] * cnn2_shape[2] * cnn2_shape[3]

        out = self.fc_layer(tf.reshape(cnn2, [-1, nodes]), 128, "output")
        return out

    def fc_layer(self, tensor, n_weight, name):
        n_prev_weight = tensor.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name + 'b', dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(tensor, W), b)
        return fc

    def cnn_layer(self, tensor, name, shape):
        conv_weights = tf.get_variable(name + "weight", shape,
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_biases = tf.get_variable(name + "bias", [shape[3]], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(tensor, conv_weights, strides=[1, 1, 1, 1], padding="VALID")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        return pool

    def cosine(self, vec1, vec2):
        p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
        return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)), tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss(self):
        margin = 0.3
        loss = tf.maximum(0.0, margin + self.d_pos - self.d_neg)
        return tf.reduce_mean(loss)

    def cal_accu(self):
        mean = tf.cast(tf.argmin(tf.stack([self.d_neg, self.d_pos], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)
