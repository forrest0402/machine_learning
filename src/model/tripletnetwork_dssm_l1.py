# -*- coding: utf-8 -*-

import random

import tensorflow as tf

FILTER_SIZES = [2, 3, 5, 7]
FILTER_DEPTH = 100
GLOBAL_VARIABLE = dict()


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

        # with tf.name_scope("positive_cosine_similarity"):
        #     self.positive_sim = self.cosine(self.anchor_output, self.positive_output)
        #
        # with tf.name_scope("negative_cosine_similarity"):
        #     self.negative_sim = self.cosine(self.anchor_output, self.negative_output)

        with tf.name_scope("l1_distance"):
            self.post_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.positive_output))
            self.neg_sim_l1 = tf.negative(self.l1norm(self.anchor_output, self.negative_output))

        with tf.name_scope("loss"):
            self.loss = self.cal_loss_l1()
            if regularizer is not None:
                self.loss += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                # tf.losses.get_regularization_loss()  # tf.add_n(tf.get_collection("losses"))

        with tf.name_scope("accuracy"):
            self.accuracy = self.cal_accu_l1()

    def network(self, x, name, reuse=True):
        filter_list = []
        for i, filter_size in enumerate(FILTER_SIZES):
            with tf.variable_scope("filter_{}".format(filter_size)):
                cnn = tf.layers.conv2d(x, filters=FILTER_DEPTH, kernel_size=[filter_size, x.shape[2].value],
                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                       reuse=reuse, use_bias=False, name="cnn")

                if "cnn_{}_bn".format(i) not in GLOBAL_VARIABLE:
                    bn, mean, var, beta, offset = self.batch_normalization(cnn, FILTER_DEPTH)
                    GLOBAL_VARIABLE["cnn_{}_bn".format(i)] = (mean, var, beta, offset)
                else:
                    bn = self.batch_same(cnn, GLOBAL_VARIABLE["cnn_{}_bn".format(i)])

                pool = tf.layers.max_pooling2d(tf.nn.tanh(bn), [bn.shape[1].value, bn.shape[2].value],
                                               [1, 1], name="pool-{}".format(name))

                filter_list.append(pool)
                if not reuse:
                    print("cnn: {} pool: {}".format(cnn.shape, pool.shape))

        flatten = tf.layers.flatten(tf.concat(filter_list, axis=3), 'flatten_layer')
        print("flatten shape: {}".format(flatten.shape))

        output_fcl = tf.layers.dense(flatten, FILTER_DEPTH, name="fcl1", use_bias=True,
                                     reuse=reuse, kernel_regularizer=self.regularizer,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.1))

        if not reuse:
            output_bn, mean, var, beta, offset = self.batch_normalization(output_fcl, FILTER_DEPTH)
            GLOBAL_VARIABLE["output_bn"] = (mean, var, beta, offset)
        else:
            output_bn = self.batch_same(output_fcl, GLOBAL_VARIABLE["output_bn"])

        out = tf.layers.dropout(tf.nn.tanh(output_bn), training=self.training, name="output_dropout-{}".format(name))
        return out

    def l1norm(self, vec1, vec2):
        return tf.reduce_sum(tf.abs(vec1 - vec2), axis=1)

    # def cosine(self, vec1, vec2):
    #     p = tf.reduce_sum(tf.multiply(vec1, vec2), axis=1)
    #     return p / tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vec1), 1)),
    #                            tf.sqrt(tf.reduce_sum(tf.square(vec2), 1)))

    def cal_loss_l1(self):
        batch_size = 128
        if random.random() >= 0.5:
            logits = tf.nn.softmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1))
            labels = tf.constant([1] * batch_size, shape=[batch_size, 1], dtype=tf.float32)
        else:
            logits = tf.nn.softmax(tf.stack([self.post_sim_l1, self.neg_sim_l1], axis=1))
            labels = tf.constant([0] * batch_size, shape=[batch_size, 1], dtype=tf.float32)
        right = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))

        loss = tf.multiply(tf.subtract(1.0, labels), tf.square(right)) + \
               tf.multiply(labels, tf.square(tf.maximum(tf.subtract(1.0, right), 0.0)))
        return tf.reduce_mean(loss)

    def cal_accu_l1(self):
        mean = tf.cast(tf.argmax(tf.stack([self.neg_sim_l1, self.post_sim_l1], axis=1), axis=1), dtype=tf.float32)
        return tf.reduce_mean(mean)

    def batch_normalization(self, conv, num_filters, need_test=True):
        beta = tf.Variable(tf.ones(num_filters), name='beta')
        offset = tf.Variable(tf.zeros(num_filters), name='offset')
        bnepsilon = 1e-5

        axis = list(range(len(conv.shape) - 1))
        batch_mean, batch_var = tf.nn.moments(conv, axis)
        ema = tf.train.ExponentialMovingAverage(0.99)
        update_ema = ema.apply([batch_mean, batch_var])

        def mean_var_with_update(batch_mean, batch_var):
            with tf.control_dependencies([update_ema]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.equal(need_test, True),
                            lambda: mean_var_with_update(ema.average(batch_mean), ema.average(batch_var)),
                            lambda: (batch_mean, batch_var))

        Ybn = tf.nn.batch_normalization(conv, mean, var, offset, beta, bnepsilon)

        return Ybn, mean, var, beta, offset

    def batch_same(self, conv, args):
        bnepsilon = 1e-5
        Ybn = tf.nn.batch_normalization(conv, args[0], args[1], args[2], args[3], bnepsilon)
        return Ybn
