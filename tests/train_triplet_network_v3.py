# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/8/30 下午1:59
"""

import sys
import os
import numpy as np
import tensorflow as tf

sys.path.extend([os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

import src.utils.converter as converter
from src.model.tripletnetwork_v3 import TripletNetwork

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 128
EPOCH = 5
BUFFER_SIZE = 1024
FILE_LINE_NUM = 0
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

train_file_name = os.path.join(ROOT_PATH, 'data/train_tokenize.txt')
test_file_name = os.path.join(ROOT_PATH, 'data/test_tokenize.txt')
word2vec_file_name = os.path.join(ROOT_PATH, 'data/model.vec')
model_save_path = os.path.join(ROOT_PATH, 'model/')
model_name = "triplet_network.ckpt"


def train():
    """

    :param train_data:
    :return:
    """
    with tf.Graph().as_default() as g:
        train_data = make_dataset(train_file_name)
        iterator = train_data.make_initializable_iterator()
        input_element = iterator.get_next()

        id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                     for index, line in enumerate(open(word2vec_file_name, 'r'))}
        id2vector[-1] = [0.0] * 256
        id2vector[-2] = [1.0] * 256

        model = TripletNetwork(25, 256, train=True)
        global_step = tf.Variable(0.0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step,
                                                   decay_steps=5000,
                                                   decay_rate=0.1, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(0.001).minimize(model.loss, global_step=global_step)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        saver = tf.train.Saver(max_to_keep=10)
        with tf.Session(config=tf_config) as sess:

            sess.run(iterator.initializer)
            # if os.path.exists(model_save_path + model_name + '.meta'):
            #     saver = tf.train.import_meta_graph(model_save_path + model_name + '.meta')
            #     saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

            ckpt = tf.train.get_checkpoint_state(model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

            for epoch_num in range(EPOCH):
                for step in range(int(FILE_LINE_NUM / BATCH_SIZE)):

                    input = sess.run(input_element)
                    x1, x2, x3 = converter.convert_input(input, id2vector)

                    _, loss_v, accu, __ = sess.run(
                        [train_op, model.loss, model.accuracy, global_step],
                        feed_dict={
                            model.anchor_input: x1,
                            model.positive_input: x2,
                            model.negative_input: x3})

                    if step % 100 == 0:
                        print("epoch {}, step {}/{}: loss {} accuracy {}"
                              .format(epoch_num, step, int(FILE_LINE_NUM / BATCH_SIZE),
                                      loss_v, accu))
                        saver.save(sess, model_save_path + model_name, global_step=global_step)

                    if np.isnan(loss_v):
                        print('Model diverged with loss = NaN')
                        quit()

                saver.save(sess, model_save_path + model_name, global_step=global_step)
                print('step %d: loss %.3f' % (step, loss_v))

                # output_grap_def = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=[''])


def make_dataset(file_name):
    """

    :param file_name:
    :return:
    """
    return tf.data.TextLineDataset(file_name) \
        .map(lambda s: tf.string_split([s], delimiter="\t").values) \
        .shuffle(buffer_size=BUFFER_SIZE) \
        .batch(batch_size=BATCH_SIZE).repeat(EPOCH)


def main(argv=None):
    """

    :param argv:
    :return:
    """

    print("************** start *****************")
    p = os.popen('wc -l {}'.format(train_file_name))
    global FILE_LINE_NUM
    FILE_LINE_NUM = int(p.read().strip().split(' ')[0])
    print("Input file has {} lines".format(FILE_LINE_NUM))
    train()


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.app.run()
