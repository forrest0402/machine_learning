# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/8/20 下午2:54
"""
import numpy as np


def convert_train(file_name):
    with open(file_name, "r", encoding="utf-8") as fr:
        with open(file_name.replace(".txt", "_tokenize.txt"), 'w', encoding="utf-8") as fw:
            for index, line in enumerate(fr.readlines()):
                if index % 100000 == 0:
                    fw.flush()
                    print("read {} lines".format(index))
                try:
                    array = line.split('\t')
                    labels = list(map(int, array[0].split(' ')))
                    if labels[1] == labels[0]:
                        fw.write("{}\t{}\t{}".format(array[2], array[1], array[3]))
                    if labels[1] == labels[2]:
                        fw.write("{}\t{}\t{}".format(array[2], array[3], array[1]))
                    # fw.write('\n')
                except Exception:
                    print(line)


if __name__ == '__main__':
    # convert_train('../data/train.txt')
    # convert_train('../data/tests.txt')
    with open('../data/test_tokenize.txt', 'r', encoding='utf-8') as fr:
        test_data = []
        for line in fr.readlines():
            test_data.append([list(map(lambda x: [x], line.split('\t')))])
        test = np.array(test_data)
        print(test.shape)

    with open('../data/train_tokenizewrong.txt', "r", encoding="utf-8") as fr:
        with open('../data/train_tokenize.txt', 'w', encoding="utf-8") as fw:
            for line in fr.readlines():
                array = line.split('\t')
                flag = False
                if len(array) != 3:
                    flag = True
                    continue
                for w in array:
                    if len(w.split(' ')) != 25:
                        flag = True
                        break
                if not flag:
                    fw.write(line)
