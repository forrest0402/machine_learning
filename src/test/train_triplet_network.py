# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from src.model.tripletnetwork import TripletNetwork

BATCH_SIZE = 8
EPOCH = 10
BUFFER_SIZE = 64
FILE_LINE_NUM = 17077358
file_name = '../data/faq_training_tokenize.txt'


def generator():
    with open(file_name, 'r', encoding="utf-8")as f:
        for line in f.readlines():
            yield [x for x in line.split('\t')]


def get_ebedding(input, embedding_matrix):
    return tf.nn.embedding_lookup(embedding_matrix, input)


def train(train_data):
    id2vector = dict()
    for index, line in enumerate(open('../data/model.vec', 'r', encoding="utf-8")):
        word_embedding = line.split(' ')
        if len(word_embedding) >= 256:
            id2vector[index - 1] = list(map(float, word_embedding[1:]))
    id2vector[-1] = [0] * 256

    model = TripletNetwork(25, 256)
    iterator = train_data.make_initializable_iterator()
    train_step = tf.train.AdamOptimizer(0.01).minimize(model.loss)
    saver = tf.train.Saver()
    input_element = iterator.get_next()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer)
        for epoch_num in range(EPOCH):
            for step in range(int(FILE_LINE_NUM / BATCH_SIZE)):
                input = sess.run(input_element)
                print(np.array(list(map(lambda x: str(x).split(' '), input[:, 0]))))
                label = np.array(list(map(lambda x: str(x).split(' '), input[:, 0]))).astype(np.float)
                input1 = np.array(list(map(lambda x: str(x).split(' '), input[:, 1]))).astype(np.int32)
                input2 = np.array(list(map(lambda x: str(x).split(' '), input[:, 2]))).astype(np.int32)
                input3 = np.array(list(map(lambda x: str(x).split(' '), input[:, 3]))).astype(np.int32)
                print(input1.shape)
                x1 = get_ebedding(input1, id2vector)
                x2 = get_ebedding(input2, id2vector)
                x3 = get_ebedding(input3, id2vector)
                _, loss_v = sess.run([train_step, model.loss], feed_dict={
                    model.x1: x1,
                    model.x2: x2,
                    model.x3: x3,
                    model.y_: label})

                if step % 1000 == 0:
                    print("epoch {}: step {} accuracy{}".format(epoch_num, loss_v, model.accuracy))

                if np.isnan(loss_v):
                    print('Model diverged with loss = NaN')
                    quit()

                if step % 10000 == 0:
                    saver.save(sess, './model/triplet_network.pb')
                    print('step %d: loss %.3f' % (step, loss_v))


def main(argv=None):
    dataset = tf.data.Dataset().from_generator(generator, output_types=tf.string) \
        .shuffle(buffer_size=BUFFER_SIZE) \
        .batch(batch_size=BATCH_SIZE).repeat(EPOCH)
    # p = os.popen('wc -l {}'.format(file_name))
    # FILE_LINE_NUM = int(p.read().split(' ')[0])
    FILE_LINE_NUM = 17077358
    print("Input file has {} lines".format(FILE_LINE_NUM))
    train(dataset)


if __name__ == '__main__':
    tf.app.run()
