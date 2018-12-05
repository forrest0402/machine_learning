# -*- coding: utf-8 -*-
import tensorflow as tf


class PlainCNN:

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
        flatten = tf.layers.flatten(input_tensor)
        output = tf.layers.dense(flatten, flatten.shape[1].value // 2, name="input", reuse=reuse)
        output = tf.reshape(output, [-1, input_tensor.shape[1].value, input_tensor.shape[2].value // 2, 1])
        if not reuse:
            print("output shape {}".format(output.shape))
        output = tf.layers.conv2d(output, 512, [3, output.shape[2].value], name="cnn1", reuse=reuse,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.01))
        output = tf.layers.max_pooling2d(output, [2, 1], [1, 1])
        if not reuse:
            print("output shape {}".format(output.shape))
        output = tf.layers.conv2d(output, 1024, [5, 1], name="cnn2", reuse=reuse, activation=tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  bias_initializer=tf.constant_initializer(0.01))
        output = tf.layers.max_pooling2d(output, [2, 1], [1, 1])
        if not reuse:
            print("output shape {}".format(output.shape))

        flatten = tf.layers.flatten(output)
        if not reuse:
            print("flatten shape {}".format(flatten.shape))
        fcl = tf.layers.dense(flatten, 128, name="fcl", reuse=reuse)
        return fcl

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
