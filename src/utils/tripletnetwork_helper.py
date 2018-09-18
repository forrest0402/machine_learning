# -*- coding: utf-8 -*-
"""
@Author: xiezizhe 
@Date: 2018/9/5 下午5:36
"""
import os

import numpy as np

import control

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
word2vec_file_name = os.path.join(ROOT_PATH, 'data/model.vec')


def get_id_vector():
    """

    Returns: dictionary whose key is word id, value is its word embedding

    """
    print("read read word embedding")
    if control.high_version(flag=True):
        id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                     for index, line in enumerate(open(word2vec_file_name, 'r', encoding="utf-8"))}
    else:
        id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                     for index, line in enumerate(open(word2vec_file_name, 'r'))}

    # EOS
    id2vector[-1] = [1.0] * 256

    # UNK
    id2vector[-2] = [0.0] * 256

    return id2vector


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
    if control.high_version():
        input1 = np.array(
            list(map(lambda x: str(x, encoding="utf-8").split(' '), input[:, 0]))).astype(
            np.int32)
        input2 = np.array(
            list(map(lambda x: str(x, encoding='utf-8').split(' '), input[:, 1]))).astype(
            np.int32)
        input3 = np.array(
            list(map(lambda x: str(x, encoding='utf-8').split(' '), input[:, 2]))).astype(
            np.int32)
        x1 = np.array(get_ebedding(input1, id2vector))
        x2 = np.array(get_ebedding(input2, id2vector))
        x3 = np.array(get_ebedding(input3, id2vector))
    else:
        input1 = np.array(
            list(map(lambda x: str(x).split(' '), input[:, 0]))).astype(
            np.int32)
        input2 = np.array(
            list(map(lambda x: str(x).split(' '), input[:, 1]))).astype(
            np.int32)
        input3 = np.array(
            list(map(lambda x: str(x).split(' '), input[:, 2]))).astype(
            np.int32)
        x1 = np.array(get_ebedding(input1, id2vector))
        x2 = np.array(get_ebedding(input2, id2vector))
        x3 = np.array(get_ebedding(input3, id2vector))
    return x1, x2, x3


def count_line_number():
    pass


def write_to_log(file_name, summary_writer, graph, ):
    write = summary_writer(file_name, graph)
    write.close()


def write_loss(file_name, loss):
    """

    Args:
        file_name:
        loss:

    Returns:

    """
    if control.high_version():
        with open(file_name, 'a', encoding="utf-8") as f:
            f.write(str(loss))
            f.write('\n')
    else:
        with open(file_name, 'a') as f:
            f.write(loss)
            f.write('\n')
