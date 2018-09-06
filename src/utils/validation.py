# -*- coding: utf-8 -*-
"""
验证生成的train_tokenize.txt格式是否正确
@Author: xiezizhe
@Date: 2018/9/6 下午5:17
"""


def validate(file_name):
    with open(file_name, 'r') as fr:
        for idx, line in enumerate(fr.readlines()):
            try:
                data = line.split('\t')
                flag = 0
                if len(data) == 3:
                    for i in range(0, 3):
                        if len(data[i].split(' ')) != 25:
                            flag = 1
                            break
                else:
                    flag = 2

                if flag == 1:
                    print("length is not 25: {}: {}".format(idx, line))
                if flag == 2:
                    print("length is not 3: {}: {}".format(idx, line))
            except Exception:
                print("invalid line data {}: {}".format(idx, line))


if __name__ == '__main__':
    print("*******************validate train.txt*******************")
    validate('train_tokenize.txt')
    print("*******************validate test.txt*******************")
    validate('test_tokenize.txt')
