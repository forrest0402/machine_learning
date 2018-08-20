# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/8/20 下午2:54
"""


def get_embedding(word2idx, sentence):
    embedding = []
    for term in sentence.split(' '):
        if term in word2idx:
            embedding.append(word2idx[term])
        else:
            embedding.append(-2)
    if len(embedding) > 25:
        embedding = embedding[:24]
    while len(embedding) < 25:
        embedding.append(-1)
    value = " ".join(map(str, embedding))
    return value


def convert_train(file_name):
    word2idx = {line.split(' ')[0]: index - 1 for index, line in
                enumerate(open('../data/model.vec', 'r'))}

    print("{} lines in total".format(len(word2idx)))

    with open(file_name, "r") as fr:
        with open(file_name.replace(".txt", "_tokenize.txt"), 'w') as fw:
            for index, line in enumerate(fr.readlines()):
                if index % 100000 == 0:
                    fw.flush()
                    print("read {} lines".format(index))
                try:
                    array = line.split('\t')
                    if "1 0" == array[0]:
                        fw.write("{}\t{}\t{}".format(get_embedding(word2idx, array[2]),
                                                     get_embedding(word2idx, array[1]),
                                                     get_embedding(word2idx, array[3])))
                    if "0 1" == array[0]:
                        fw.write("{}\t{}\t{}".format(get_embedding(word2idx, array[2]),
                                                     get_embedding(word2idx, array[3]),
                                                     get_embedding(word2idx, array[1])))
                    fw.write('\n')
                except Exception:
                    print(line)


if __name__ == '__main__':
    convert_train('../data/train.txt')
    convert_train('../data/test.txt')
