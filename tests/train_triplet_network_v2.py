# -*- coding: utf-8 -*-
import sys
import os

sys.path.extend([os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

import numpy as np
import tensorflow as tf

from src.model.tripletnetwork_v2 import TripletNetwork

BATCH_SIZE = 128
EPOCH = 5
BUFFER_SIZE = 1024
FILE_LINE_NUM = 0
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
train_file_name = os.path.join(ROOT_PATH, 'data/train_tokenize.txt')
test_file_name = os.path.join(ROOT_PATH, 'data/test_tokenize.txt')
word2vec_file_name = os.path.join(ROOT_PATH, 'data/model.vec')


def get_ebedding(input, embedding_matrix):
    # return tf.nn.embedding_lookup(embedding_matrix, input)
    return [[embedding_matrix[id] for id in one] for one in input]


def convert_input(input, id2vector):
    input1 = np.array(list(map(lambda x: str(x).split(' '), input[:, 0]))).astype(
        np.int32)
    input2 = np.array(list(map(lambda x: str(x).split(' '), input[:, 1]))).astype(
        np.int32)
    input3 = np.array(list(map(lambda x: str(x).split(' '), input[:, 2]))).astype(
        np.int32)
    x1 = np.array(get_ebedding(input1, id2vector))
    x2 = np.array(get_ebedding(input2, id2vector))
    x3 = np.array(get_ebedding(input3, id2vector))
    return x1, x2, x3


def test(sess, model, id2vector):
    print("************************** tests *******************************")
    with open(test_file_name, 'r') as fr:
        test_data = []
        for line in fr.readlines():
            test_data.append(line.split('\t'))
        test_data = np.array(test_data)
        print(test_data.shape)
        x1, x2, x3 = convert_input(test_data, id2vector)

        accu = sess.run([model.accuracy],
                        feed_dict={
                            model.anchor_input: x1,
                            model.positive_input: x2,
                            model.negative_input: x3})
        print("tests accu {}".format(accu))


def train(train_data):
    id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                 for index, line in enumerate(open(word2vec_file_name, 'r'))}
    id2vector[-1] = [0.0] * 256
    id2vector[-2] = [1.0] * 256

    model = TripletNetwork(25, 256)
    iterator = train_data.make_initializable_iterator()
    train_step = tf.train.AdamOptimizer(0.001).minimize(model.loss)
    saver = tf.train.Saver(max_to_keep=5)
    input_element = iterator.get_next()

    sess_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess_conf.gpu_options.allow_growth = True

    model_save_path = './model/triplet_network.ckpt'
    with tf.Session(config=sess_conf) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(iterator.initializer)
        if os.path.exists(model_save_path + '.meta'):
            saver = tf.train.import_meta_graph(model_save_path + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint("./model/"))
            test(sess, model, id2vector)
        # tests(sess, model, id2vector)
        for epoch_num in range(EPOCH):
            for step in range(int(FILE_LINE_NUM / BATCH_SIZE)):
                input = sess.run(input_element)
                x1, x2, x3 = convert_input(input, id2vector)

                _, loss_v, accu, anchor, pos, neg = sess.run(
                    [train_step, model.loss, model.accuracy, model.anchor_output, model.d_pos,
                     model.d_neg],
                    feed_dict={
                        model.anchor_input: x1,
                        model.positive_input: x2,
                        model.negative_input: x3})

                if step % 1000 == 0:
                    # print(anchor)
                    # print(pos)
                    # print(neg)
                    print("epoch {}, step {}/{}: loss {} accuracy {}"
                          .format(epoch_num, step, int(FILE_LINE_NUM / BATCH_SIZE), loss_v, accu))

                if np.isnan(loss_v):
                    print('Model diverged with loss = NaN')
                    quit()

                if step % 10000 == 0:
                    saver.save(sess, model_save_path)
                    print('step %d: loss %.3f' % (step, loss_v))
                    test(sess, model, id2vector)

            # output_grap_def = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=[''])


def make_dataset(file_name):
    return tf.data.TextLineDataset(file_name) \
        .map(lambda s: tf.string_split([s], delimiter="\t").values) \
        .shuffle(buffer_size=BUFFER_SIZE) \
        .batch(batch_size=BATCH_SIZE).repeat(EPOCH)


def main(argv=None):
    print("************** start *****************")
    train_data = make_dataset(train_file_name)
    p = os.popen('wc -l {}'.format(train_file_name))
    global FILE_LINE_NUM
    FILE_LINE_NUM = int(p.read().strip().split(' ')[0])
    print("Input file has {} lines".format(FILE_LINE_NUM))
    train(train_data)


if __name__ == '__main__':
    tf.app.run()
