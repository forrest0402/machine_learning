# -*- coding: utf-8 -*-
import tensorflow as tf

import src.loss.triplet_loss as triplet_loss


class TripletNetwork:

    def __init__(self, length, embedding_size):
        self.hidden_layer_num = length * 64

        self.left = tf.placeholder(tf.float32, [None, length, embedding_size])
        self.center = tf.placeholder(tf.float32, [None, length, embedding_size])
        self.right = tf.placeholder(tf.float32, [None, length, embedding_size])

        with tf.variable_scope("triplet") as scope:
            self.o1 = self.network(self.left)
            scope.reuse_variables()
            self.o2 = self.network(self.center)
            self.o3 = self.network(self.right)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None, 3])
        self.loss = self.cal_loss()

        # Calculate precision
        predictions = tf.reshape(tf.argmin(tf.stack([self.o1, self.o3], axis=1), axis=1), [-1])
        correct_predictions = tf.equal(tf.cast(tf.equal(self.y_[:, 1], self.y_[:, 2]), dtype=tf.int64), predictions)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))

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
        return tf.reshape(cnn2, [-1, nodes])

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

    def cal_loss(self):
        labels = tf.reshape(self.y_, [-1, 1])
        embedding = tf.concat([self.o1, self.o2, self.o3], axis=0)
        margin = 0.3
        return triplet_loss.batch_hard_triplet_loss(labels, embedding, margin)
