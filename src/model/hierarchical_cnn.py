# -*- coding: utf-8 -*-

import tensorflow as tf

LOOP_SIZE = 4
FILTER_DEPTH = 128


class HierarchicalCNN:

    def __init__(self, max_length, embedding_size=128, trainable=True,
                 regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):
        self.positive_input = tf.placeholder(tf.float32, [None, max_length, embedding_size], name="positive_input")
        self.anchor_input = tf.placeholder(tf.float32, [None, max_length, embedding_size], name="anchor_input")
        self.negative_input = tf.placeholder(tf.float32, [None, max_length, embedding_size], name="negative_input")
        self.training = tf.placeholder(tf.bool, name="training")
        self.regularizer = regularizer
        self.trainable = trainable

        with tf.variable_scope("triplet"):
            self.anchor_output = self.network(tf.expand_dims(self.anchor_input, -1), "anchor", False)
            self.positive_output = self.network(tf.expand_dims(self.positive_input, -1), "pos", True)
            self.negative_output = self.network(tf.expand_dims(self.negative_input, -1), "neg", True)

        with tf.name_scope("cosine_similarity"):
            self.pos_sim = self.cosine(self.anchor_output, self.positive_output)
            self.neg_sim = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("loss"):
            self.loss = self.cal_loss1() + self.cal_loss2()
            if regularizer is not None:
                self.loss += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu()

    # (,25,128,1)
    def network(self, input, name, reuse=False):
        outputs = []
        for i in range(LOOP_SIZE):
            cnn = tf.layers.conv2d(input, filters=FILTER_DEPTH,
                                   kernel_size=[3, self.anchor_input.shape[2].value],
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   reuse=reuse, bias_initializer=tf.constant_initializer(0.1),
                                   padding="SAME",
                                   name="cnn1", trainable=self.trainable)
            pool = tf.layers.max_pooling2d(cnn, [cnn.shape[1].value, 1], [1, 1],
                                           name="pool{}".format(name))
            outputs.append(pool)
