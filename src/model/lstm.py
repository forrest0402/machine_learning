# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import sys

"""
@Author: xiezizhe 
@Date: 2018/11/15 5:15 PM
"""
sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class LSTMLM:

    def __init__(self, batch_size, vocab_size, num_hidden_units, dropout_rate):
        self.file_name_train = tf.placeholder(tf.string)
        self.file_name_validation = tf.placeholder(tf.string)
        self.file_name_test = tf.placeholder(tf.string)
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_hidden_units = num_hidden_units

        def parse(line):
            line_split = tf.string_split([line])
            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
            return input_seq, output_seq

        training_dataset = tf.data.TextLineDataset(self.file_name_train) \
            .map(parse) \
            .shuffle(256) \
            .padded_batch(self.batch_size, padded_shapes=([None], [None]))

        validation_dataset = tf.data.TextLineDataset(self.file_name_validation) \
            .map(parse) \
            .padded_batch(self.batch_size, padded_shapes=([None], [None]))

        test_dataset = tf.data.TextLineDataset(self.file_name_test).map(parse).batch(1)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                   training_dataset.output_shapes)

        self.input_batch, self.output_batch = iterator.get_next()
        self.trining_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)

        self.input_embedding_mat = tf.get_variable("input_embedding_mat", [self.vocab_size, self.num_hidden_units],
                                                   dtype=tf.float32)
        with tf.device("/cpu:0"):
            self.input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, self.input_batch)

        # Output embedding
        self.output_embedding_mat = tf.get_variable("output_embedding_mat",
                                                    [self.vocab_size, self.num_hidden_units], dtype=tf.float32)

        self.output_embedding_bias = tf.get_variable("output_embedding_bias", [self.vocab_size], dtype=tf.float32)

        self.outputs, self.valid_words, self.loss = self.network()

    def network(self):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_hidden_units, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell], state_is_tuple=True)

        non_zero_weights = tf.sign(self.input_batch)
        valid_words = tf.reduce_sum(non_zero_weights)

        # Compute sequence length
        def get_length(non_zero_place):
            real_length = tf.reduce_sum(non_zero_place, 1)
            real_length = tf.cast(real_length, tf.int32)
            return real_length

        batch_length = get_length(non_zero_weights)

        # [batch_size, max_time, embed_size]，batch_size是输入的这批数据的数量，max_time就是这批数据中序列的最长长度，embed_size表示嵌入的词向量的维度。
        outputs, _ = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=self.input_embedded,
                sequence_length=batch_length,
                dtype=tf.float32
        )

        def output_embedding(current_output):
            return tf.add(
                    tf.matmul(current_output, tf.transpose(self.output_embedding_mat), self.output_embedding_bias))

        logits = tf.map_fn(output_embedding, outputs)
        logits = tf.reshape(logits, [-1, vocab_size])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.output_batch, [-1]),
                logits=logits
        ) * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)
        return outputs, valid_words, tf.reduce_sum(loss)
