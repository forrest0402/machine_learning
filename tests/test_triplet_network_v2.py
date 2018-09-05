# -*- coding: utf-8 -*-
"""
@Author: xiezizhe 
@Date: 2018/9/5 下午4:59
"""

import sys
import os
import numpy as np
import tensorflow as tf

sys.path.extend([os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

import src.utils.converter as converter
from src.model.tripletnetwork_v2 import TripletNetwork
import tests.train_triplet_network_v2 as train

if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        print("************************** tests *******************************")
        x1 = tf.placeholder(tf.float32, [None, 25, 256])
        x2 = tf.placeholder(tf.float32, [None, 25, 256])
        x3 = tf.placeholder(tf.float32, [None, 25, 256])
        model = TripletNetwork(25, 256)

        # read test file
        print("read test file")
        test_data = []
        with open(train.test_file_name, 'r') as fr:
            # for line in fr.readlines():
            #     test_data.append(line.split('\t'))
            test_data = np.array([line.split('\t') for line in fr.readlines()], dtype=np.string_)
            # test_data = np.array(test_data)

        test_data = test_data[0:2000, :]
        print(test_data.shape)
        # read word embedding
        print("read read word embedding")
        id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                     for index, line in enumerate(open(train.word2vec_file_name, 'r'))}
        id2vector[-1] = [0.0] * 256
        id2vector[-2] = [1.0] * 256

        print("start to construct inputs")
        x1, x2, x3 = converter.convert_input(test_data, id2vector)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("start to calculate accuracy")
            ckpt = tf.train.get_checkpoint_state(train.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                accu = sess.run([model.accuracy],
                                feed_dict={
                                    model.anchor_input: x1,
                                    model.positive_input: x2,
                                    model.negative_input: x3})

                print("tests accu {}".format(accu))

            else:
                print("no saved model found")
