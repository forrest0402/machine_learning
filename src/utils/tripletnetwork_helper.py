# -*- coding: utf-8 -*-
"""
@Author: xiezizhe 
@Date: 2018/9/5 下午5:36
"""
import logging
import os
from itertools import islice

import numpy as np

import control

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
WORD2VEC_FILE_NAME = os.path.join(ROOT_PATH, 'data/model.vec')


def get_word_vec(word2vec_file_name):
    logging.info("get_word_vecs {}".format(word2vec_file_name))
    if control.high_version(flag=True):
        id2vector = [np.array(line.split(' ')[1:], dtype=np.float32)
                     for line in islice(open(word2vec_file_name, 'r', encoding="utf-8"), 1, None)]
    else:
        id2vector = [np.array(line.split(' ')[1:], dtype=np.float32)
                     for line in islice(open(word2vec_file_name, 'r'), 1, None)]

    # id2vector.append(np.random.normal(0, 1, size=(128,)))
    # id2vector.append(np.random.normal(0, 1, size=(128,)))
    id2vector.append(np.array(np.repeat(1.0, 128), dtype=np.float32))
    id2vector.append(np.zeros([128], dtype=np.float32))
    word_vec = np.array(id2vector, dtype=np.float32).reshape([len(id2vector), 128])
    logging.info("exit get_word_vecs {}".format(word_vec.shape))
    return word_vec


def get_id_vector(word2vec_file_name=None):
    """

    Returns: dictionary whose key is word id, value is its word embedding

    """
    if word2vec_file_name is None:
        word2vec_file_name = WORD2VEC_FILE_NAME

    logging.info("read word embedding from {}".format(word2vec_file_name))
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

    logging.info("exit get_id_vector")
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


def read_from_input(input):
    """
    example: input= "1,2,3 4,5,6 7,8,9", return=([1,2,3], [4,5,6], [7,8,9])

    Args:
        input:

    Returns:

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
    return input1, input2, input3


def get_input_embedding(input, id2vector):
    """

    :param input: [, 3]
    :param id2vector: [n, d] n words each word has d dimension
    :return:
    """
    input1, input2, input3 = read_from_input(input)
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
    path = os.path.split(file_name)[0]
    if not os.path.exists(path):
        os.mkdir(path)
        os.system("touch {}".format(file_name))

    if control.high_version():
        with open(file_name, 'a', encoding="utf-8") as f:
            f.write(str(loss))
            f.write('\n')
    else:
        with open(file_name, 'a') as f:
            f.write(str(loss))
            f.write('\n')
