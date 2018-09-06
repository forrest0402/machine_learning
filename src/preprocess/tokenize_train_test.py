# -*- coding: utf-8 -*-
import random
import os
from itertools import combinations

"""
将之前dssm需要的训练数据转化成triplet_network的需要的数据输入格式
@Author: xiezizhe
@Date: 2018/8/10 下午3:24
"""


def replace_str(str):
    try:
        return str.replace("\r", "").replace("?", "").replace("？", "").strip()
    except Exception:
        return str


def get_embedding(word2idx, sentence):
    embedding = []
    for term in sentence.split(' '):
        if term in word2idx:
            embedding.append(word2idx[term])
    if len(embedding) > 25:
        embedding = embedding[:24]
    while len(embedding) < 25:
        embedding.append(-1)
    value = " ".join(map(str, embedding))
    return value


def tokenize_embedding(output_file_name):
    p = os.popen('wc -l {}'.format(output_file_name))
    file_line_num = int(p.read().strip().split(' ')[0])
    print("Input file has {} lines".format(file_line_num))

    word2idx = dict()
    total_num = 0
    for index, line in enumerate(open('model.vec', 'r')):
        word_embedding = line.split(' ')
        total_num += 1
        if len(word_embedding) >= 256:
            word2idx[word_embedding[0]] = index - 1
    print("{} lines in total".format(total_num))
    with open(output_file_name, "r") as fr:
        with open(output_file_name.replace(".txt", "_tokenize.txt"), 'w') as fw:
            for index, line in enumerate(fr.readlines()):
                if index % 300000 == 0:
                    print("read {} lines".format(index))
                try:
                    array = line.split('\t')
                    if "0 1" == array[0]:
                        fw.write(
                            "{}\t{}\t{}".format(get_embedding(word2idx, array[2]),
                                                get_embedding(word2idx, array[3]),
                                                get_embedding(word2idx, array[1])))
                    elif "1 0" == array[0]:
                        fw.write(
                            "{}\t{}\t{}".format(get_embedding(word2idx, array[2]),
                                                get_embedding(word2idx, array[1]),
                                                get_embedding(word2idx, array[3])))
                    else:
                        print("invalid line {}".format(line))
                    fw.write('\n')
                except Exception:
                    print(line)

    print("Exit tokenize_embedding")


if __name__ == '__main__':
    tokenize_embedding(output_file_name='train.txt')
    tokenize_embedding(output_file_name='test.txt')
    print("Hello, world")
