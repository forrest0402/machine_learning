# -*- coding: utf-8 -*-
"""
@Author: xiezizhe 
@Date: 2018/9/5 下午5:36
"""
import numpy as np


def get_ebedding(input, embedding_matrix):
    """
    word embedding
    :param input:
    :param embedding_matrix:
    :return:
    """
    # return tf.nn.embedding_lookup(embedding_matrix, input)
    return [[embedding_matrix[id] for id in one] for one in input]


def convert_input(input, id2vector):
    """

    :param input: [, 3]
    :param id2vector: [n, d] n words each word has d dimension
    :return:
    """
    input1 = np.array(list(map(lambda x: str(x).split(' '), input[:, 0]))).astype(
        np.int32)
    input2 = np.array(list(map(lambda x: str(x).split(' '), input[:, 1]))).astype(
        np.int32)
    input3 = np.array(list(map(lambda x: str(x).split(' '), input[:, 2]))).astype(
        np.int32)
    x1 = np.array(get_ebedding(input1, id2vector))
    x2 = np.array(get_ebedding(input2, id2vector))
    x3 = np.array(get_ebedding(input3, id2vector))
    return x1, x2, x3
