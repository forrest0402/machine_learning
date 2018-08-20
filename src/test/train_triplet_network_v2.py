# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os

from src.model.tripletnetwork_v2 import TripletNetwork

BATCH_SIZE = 64
EPOCH = 10
BUFFER_SIZE = 64
FILE_LINE_NUM = 0
train_file_name = '../data/train_tokenize.txt'
test_file_name = '../data/test_tokenize.txt'


def get_ebedding(input, embedding_matrix):
    # return tf.nn.embedding_lookup(embedding_matrix, input)
    return [[embedding_matrix[id] for id in one] for one in input]


def train(train_data):
    id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                 for index, line in enumerate(open('../data/model.vec', 'r'))}
    id2vector[-1] = [0.0] * 256
    id2vector[-2] = [1.0] * 256

    model = TripletNetwork(25, 256)
    iterator = train_data.make_initializable_iterator()
    train_step = tf.train.AdamOptimizer(0.001).minimize(model.loss)
    saver = tf.train.Saver()
    input_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(iterator.initializer)
        for epoch_num in range(EPOCH):
            for step in range(int(FILE_LINE_NUM / BATCH_SIZE)):
                input = sess.run(input_element)
                input1 = np.array(list(map(lambda x: str(x).split(' '), input[:, 0]))).astype(np.int32)
                input2 = np.array(list(map(lambda x: str(x).split(' '), input[:, 1]))).astype(np.int32)
                input3 = np.array(list(map(lambda x: str(x).split(' '), input[:, 2]))).astype(np.int32)
                x1 = get_ebedding(input1, id2vector)
                x2 = get_ebedding(input2, id2vector)
                x3 = get_ebedding(input3, id2vector)
                _, loss_v, accu = sess.run([train_step, model.loss, model.accuracy], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3})

                # print(input1)

                if step % 10 == 0:
                    print("epoch {}, step {}/{}: loss {} accuracy {}"
                          .format(epoch_num, step, int(FILE_LINE_NUM / BATCH_SIZE), loss_v, accu))

                if np.isnan(loss_v):
                    print('Model diverged with loss = NaN')
                    quit()

                if step % 10000 == 0:
                    saver.save(sess, './model/triplet_network.pb')
                    print('step %d: loss %.3f' % (step, loss_v))


def make_dataset(file_name):
    return tf.data.TextLineDataset(file_name) \
        .map(lambda s: tf.string_split([s], delimiter="\t").values) \
        .shuffle(buffer_size=BUFFER_SIZE) \
        .batch(batch_size=BATCH_SIZE).repeat(EPOCH)


def main(argv=None):
    train_data = make_dataset(train_file_name)
    p = os.popen('wc -l {}'.format(train_file_name))
    global FILE_LINE_NUM
    FILE_LINE_NUM = int(p.read().strip().split(' ')[0])
    print("Input file has {} lines".format(FILE_LINE_NUM))
    train(train_data)


if __name__ == '__main__':
    tf.app.run()
