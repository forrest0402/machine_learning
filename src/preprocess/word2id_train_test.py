# -*- coding: utf-8 -*-
import os

import sys

"""
将之前dssm需要的训练数据转化成triplet_network的需要的数据输入格式
@Author: xiezizhe
@Date: 2018/8/10 下午3:24
"""
WORD_VEC_FILE_NAME = "wordvec.vec"
SAVE_FILE = "word2id.txt"
VERSION = sys.version.split(" ")[0]
EOS_LINE_NUM = -2
UNK_LINE_NUM = -1
MAX_LEN = 32


def replace_str(str):
    try:
        return str.replace("\r", "").replace("?", "").replace("？", "").strip()
    except Exception:
        return str


def get_word_id(word2idx, sentence):
    embedding = []
    for term in sentence.split(' '):
        if term in word2idx:
            embedding.append(word2idx[term])
        else:
            embedding.append(UNK_LINE_NUM)
    if len(embedding) > MAX_LEN:
        embedding = embedding[:MAX_LEN - 1]

    if len(embedding) < MAX_LEN:
        embedding.extend([EOS_LINE_NUM] * (MAX_LEN - len(embedding)))

    # while len(embedding) < 25:
    #     embedding.append(-1)
    value = " ".join(map(str, embedding))
    return value


def tokenize_embedding(output_file_name, word2idx):
    p = os.popen('wc -l {}'.format(output_file_name))
    file_line_num = int(p.read().strip().split(' ')[0])
    print("Input file has {} lines".format(file_line_num))

    global EOS_LINE_NUM
    EOS_LINE_NUM = len(word2idx) + 1
    global UNK_LINE_NUM
    UNK_LINE_NUM = len(word2idx)

    if high_version():
        fr = open(output_file_name, "r", encoding="utf-8")
        fw = open(output_file_name.replace(".txt", "_id.txt"), 'w', encoding="utf-8", buffering=1)
    else:
        fr = open(output_file_name, "r")
        fw = open(output_file_name.replace(".txt", "_id.txt"), 'w', buffering=1)

    for index, line in enumerate(fr.readlines()):
        if index % (file_line_num // 10) == 0:
            print("read {} lines".format(index))
        if index % 10000 == 0:
            fw.flush()
        try:
            array = line.split('\t')
            if "0 1" == array[0]:
                fw.write("{}\t{}\t{}".format(get_word_id(word2idx, array[2]),
                                             get_word_id(word2idx, array[3]),
                                             get_word_id(word2idx, array[1])))
            elif "1 0" == array[0]:
                fw.write("{}\t{}\t{}".format(get_word_id(word2idx, array[2]),
                                             get_word_id(word2idx, array[1]),
                                             get_word_id(word2idx, array[3])))
            else:
                print("invalid line {}".format(line))
            fw.write('\n')
        except Exception as e:
            print(line, e)
    fr.close()
    fw.close()
    print("Exit tokenize_embedding")


def high_version(version=VERSION, flag=False):
    if flag:
        print("python version: " + version)
    if version[0] == "3":
        return True
    if version[0] == "2":
        return False

    raise RuntimeError('unkown version')


if __name__ == '__main__':

    if not os.path.isfile(WORD_VEC_FILE_NAME) and not os.path.isfile(SAVE_FILE):
        print("{} doesn't exist".format(WORD_VEC_FILE_NAME))
        quit()

    if not os.path.isfile(SAVE_FILE):
        word2idx = dict()
        total_num = 0
        first_line = True
        if high_version():
            with open(SAVE_FILE, 'w', encoding="utf-8") as fw:
                with open(WORD_VEC_FILE_NAME, "r", encoding="utf-8") as fr:
                    for index, line in enumerate(fr.readlines()):
                        word_embedding = line.split(' ')
                        total_num += 1
                        if len(word_embedding) >= 10:
                            word2idx[word_embedding[0]] = index - 1
                            if first_line:
                                fw.write("{} {}".format(word_embedding[0], index - 1))
                                first_line = False
                            else:
                                fw.write('\n')
                                fw.write("{} {}".format(word_embedding[0], index - 1))

        else:
            with open(SAVE_FILE, 'w') as fw:
                with open(WORD_VEC_FILE_NAME, "r") as fr:
                    for index, line in enumerate(fr.readlines()):
                        word_embedding = line.split(' ')
                        total_num += 1
                        if len(word_embedding) >= 10:
                            word2idx[word_embedding[0]] = index - 1
                            if first_line:
                                fw.write("{} {}".format(word_embedding[0], index - 1))
                                first_line = False
                            else:
                                fw.write('\n')
                                fw.write("{} {}".format(word_embedding[0], index - 1))
        print("{} lines in total".format(total_num))
    else:
        if high_version():
            word2idx = {line.split(" ")[0]: int(line.split(" ")[1])
                        for index, line in enumerate(open(SAVE_FILE, 'r', encoding="utf-8"))}
        else:
            word2idx = {line.split(" ")[0]: int(line.split(" ")[1])
                        for index, line in enumerate(open(SAVE_FILE, 'r'))}

    tokenize_embedding(output_file_name='train.txt', word2idx=word2idx)
    tokenize_embedding(output_file_name='test.txt', word2idx=word2idx)
    print("Hello, world")
