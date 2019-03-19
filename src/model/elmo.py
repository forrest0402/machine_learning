# -*- coding: utf-8 -*-

import tensorflow as tf
import random

"""
@Author: xiezizhe 
@Date: 2018/11/15 5:15 PM
"""


class Elmo:

    def __init__(self, vocab_size, emb_size, batch_size, lstm_size):
        self.input = tf.placeholder(tf.int32, [None, batch_size, vocab_size])

        with tf.device("/cpu:0"):
            embed_input = tf.nn.embedding_lookup(self.embedding_weight, data["input"])
