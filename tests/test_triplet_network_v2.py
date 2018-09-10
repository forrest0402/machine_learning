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
import train_triplet_network_v2 as train

BATCH_SIZE = 8192


def make_dataset(file_name):
    """

    :param file_name:
    :return:
    """
    return tf.data.TextLineDataset(file_name) \
        .map(lambda s: tf.string_split([s], delimiter="\t").values) \
        .batch(batch_size=BATCH_SIZE)


if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        print("************************** tests *******************************")

        p = os.popen('wc -l {}'.format(train.test_file_name))
        file_line_num = int(p.read().strip().split(' ')[0])
        print("Test file has {} lines".format(file_line_num))

        x1 = tf.placeholder(tf.float32, [None, 25, 256])
        x2 = tf.placeholder(tf.float32, [None, 25, 256])
        x3 = tf.placeholder(tf.float32, [None, 25, 256])
        model = TripletNetwork(25, 256)

        # read test file
        print("read test file")
        # test_data = []
        # with open(train.test_file_name, 'r') as fr:
        #     test_data = np.array([line.split('\t') for line in fr.readlines()], dtype=np.string_)
        #
        # test_data = test_data[0:2000, :]
        # print(test_data.shape)

        test_data = make_dataset(train.test_file_name)
        iterator = test_data.make_initializable_iterator()
        input_element = iterator.get_next()

        # read word embedding
        print("read read word embedding")
        id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                     for index, line in enumerate(open(train.word2vec_file_name, 'r'))}
        id2vector[-1] = [0.0] * 256
        id2vector[-2] = [1.0] * 256

        # print("start to construct inputs")
        # x1, x2, x3 = converter.convert_input(test_data, id2vector)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True
        with tf.Session(config=tf_config) as sess:

            print("start to calculate accuracy")
            sess.run(iterator.initializer)

            ckpt = tf.train.get_checkpoint_state(train.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                accus = []
                try:
                    step = 0
                    while True:
                        input = sess.run(input_element)
                        x1, x2, x3 = converter.convert_input(input, id2vector)
                        accu = sess.run([model.accuracy],
                                        feed_dict={
                                            model.anchor_input: x1,
                                            model.positive_input: x2,
                                            model.negative_input: x3})
                        accus.append(accu)
                        print("{}/{},\taccu {}, average accuracy: {}".format(step * BATCH_SIZE,
                                                                             file_line_num, accu,
                                                                             np.mean(accus)))
                        step += 1
                except tf.errors.OutOfRangeError:
                    print("end!")
                    print("test accuracy: {}".format(np.mean(accus)))

            else:
                print("no saved model found")
