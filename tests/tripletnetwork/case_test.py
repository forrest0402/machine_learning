# -*- coding: utf-8 -*-
"""
@Author: xiezizhe 
@Date: 2018/9/5 下午4:59
"""

import sys
import os
import numpy as np
import tensorflow as tf
from pyhanlp import HanLP

sys.path.extend([os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

reload(sys)
sys.setdefaultencoding('utf8')

from src.model.tripletnetwork_v4 import TripletNetwork
import train_v4 as train

BATCH_SIZE = 128


def get_embedding(sentence, word2vector):
    embedding = []
    result = []
    for term in HanLP.segment(sentence):
        result.append(term.word)

    for word in result:
        if str(word) in word2vector:
            embedding.append(word2vector[str(word)])
        else:
            print("{}, {}".format(word, len(word2vector)))
    if len(embedding) > 25:
        embedding = embedding[:24]
    while len(embedding) < 25:
        embedding.append([1.0] * 256)
    return embedding


if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        print("************************** case study *******************************")

        x1 = tf.placeholder(tf.float32, [None, 25, 256])
        x2 = tf.placeholder(tf.float32, [None, 25, 256])
        x3 = tf.placeholder(tf.float32, [None, 25, 256])
        model = TripletNetwork(25, 256)

        # read word embedding
        print("read read word embedding")
        word2vector = dict()
        for index, line in enumerate(open(train.word2vec_file_name, 'rb')):
            word_embedding = line.split(' ')
            if len(word_embedding) >= 256:
                word2vector[str(word_embedding[0])] = list(map(float, word_embedding[1:]))
            if len(word2vector) == 1000000:
                break

        temp1 = "我搜索我的订单为什么搜索不了？"
        temp2 = "我搜索我朋友给我的一个商品单号我为什么搜索不出来？"
        temp3 = "为什么我刷不了卡"

        temp1 = "为什么我朋友拨打不通你们银行的电话呢？"
        temp2 = "为什么电话那么久没人接"
        temp3 = "你们是不是微信上可以留言给我回电话的"

        # print("start to construct inputs")
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        with tf.Session(config=tf_config) as sess:

            print("start to calculate accuracy")

            ckpt = tf.train.get_checkpoint_state(train.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

                try:
                    if True:
                        x1, x2, x3 = [get_embedding(temp1, word2vector)], \
                                     [get_embedding(temp2, word2vector)], \
                                     [get_embedding(temp3, word2vector)]

                        accu, neg, pos = sess.run([model.accuracy, model.negative_sim, model.positive_sim],
                                                  feed_dict={
                                                      model.anchor_input: x1,
                                                      model.positive_input: x1,
                                                      model.negative_input: x1,
                                                      model.training: False})
                        print("accu: {}, neg distance: {}, positive distance: {}"
                              .format(accu, neg, pos))
                        print("hello, world")
                except tf.errors.OutOfRangeError:
                    print("end!")
                    print("test accuracy: {}".format(np.mean(accus)))

            else:
                print("no saved model found")
