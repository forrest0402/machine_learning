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
from src.model.tripletnetwork_v4 import TripletNetwork

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 128
EPOCH = 5
BUFFER_SIZE = 8196
FILE_LINE_NUM = 0
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

train_file_name = os.path.join(ROOT_PATH, 'data/train_tokenize.txt')
test_file_name = os.path.join(ROOT_PATH, 'data/test_tokenize.txt')
word2vec_file_name = os.path.join(ROOT_PATH, 'data/model.vec')
model_save_path = os.path.join(ROOT_PATH, 'model/')
model_name = "triplet_network.ckpt"
log_file = os.path.join(ROOT_PATH, 'log/triplet_network_v4')


def train():
    """

    :param train_data:
    :return:
    """
    train_data = make_dataset(train_file_name)
    iterator = train_data.make_initializable_iterator()
    input_element = iterator.get_next()

    id2vector = {index - 1: list(map(float, line.split(' ')[1:]))
                 for index, line in enumerate(open(word2vec_file_name, 'r'))}
    id2vector[-1] = [0.0] * 256
    id2vector[-2] = [1.0] * 256

    model = TripletNetwork(25, 256)

    global_step = tf.Variable(0.0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001,
                                               global_step=global_step,
                                               decay_steps=10000, decay_rate=0.7,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    sess = tf.InteractiveSession(config=tf_config)
    saver = tf.train.Saver(max_to_keep=5)

    sess.run(iterator.initializer)
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    print([x.name for x in tf.global_variables()])
    write = tf.summary.FileWriter(log_file, tf.get_default_graph())
    write.close()

    round_number = int(FILE_LINE_NUM / BATCH_SIZE)
    for epoch_num in range(int(global_step.eval()) / round_number, EPOCH):
        for step in range(int(global_step.eval()) % round_number, round_number):

            input = sess.run(input_element)
            x1, x2, x3 = converter.convert_input(input, id2vector)

            _, loss_v, accu, __ = sess.run(
                [train_op, model.loss, model.accuracy, global_step], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3,
                    model.training: True})

            if step % 500 == 0:
                test_accu = sess.run([model.accuracy], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3,
                    model.training: False})
                print("epoch {}, step {}/{}, loss {}, accuracy {}, test accuracy {}"
                      .format(epoch_num, step, round_number, loss_v, accu, test_accu))
                saver.save(sess, model_save_path + model_name, global_step=global_step)

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

        saver.save(sess, model_save_path + model_name, global_step=global_step)

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
