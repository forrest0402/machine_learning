# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/8/30 下午1:59
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

import src.utils.tripletnetwork_helper as helper
from src.model.tripletnetwork_dssm import TripletNetwork

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

TRAIN_BATCH_SIZE = 128
EPOCH = 5
TRAIN_BUFFER_SIZE = 8196
TRAIN_FILE_LINE_NUM = 0

TEST_BATCH_SIZE = 1024
TEST_FILE_LINE_NUM = 0

REGULARIZATION_RATE = 1e-3
MOVING_AVERAGE_DECAY = 0.99

train_file_name = os.path.join(ROOT_PATH, 'data/train_tokenize.txt')
test_file_name = os.path.join(ROOT_PATH, 'data/test_tokenize.txt')
word2vec_file_name = os.path.join(ROOT_PATH, 'data/model.vec')
model_save_path = os.path.join(ROOT_PATH, 'model_dssm/')
model_name = "triplet_network.ckpt"
log_file = os.path.join(ROOT_PATH, 'log/triplet_network_dssm')
loss_file = os.path.join(ROOT_PATH, 'loss_dssm/loss.txt')


def train():
    """

    :return:
    """
    train_data = make_dataset(train_file_name, TRAIN_BUFFER_SIZE, TRAIN_BATCH_SIZE, EPOCH)
    train_iterator = train_data.make_initializable_iterator()
    train_input_element = train_iterator.get_next()

    test_data = make_dataset(test_file_name, TEST_BATCH_SIZE, TEST_BATCH_SIZE, EPOCH)
    test_iterator = test_data.make_initializable_iterator()
    test_input_element = test_iterator.get_next()

    id2vector = helper.get_id_vector()

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    model = TripletNetwork(25, 256, regularizer)

    global_step = tf.Variable(0.0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001,
                                               global_step=global_step,
                                               decay_steps=10000, decay_rate=0.7,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    loss = model.loss + tf.add_n(tf.get_collection("losses"))
    # batch normalization need this op to update its variables
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = optimizer.minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    sess = tf.InteractiveSession(config=tf_config)
    saver = tf.train.Saver(max_to_keep=5)

    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)

    # try to load existing model
    ckpt = tf.train.get_checkpoint_state(model_save_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("load model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    print([x.name for x in tf.global_variables()])

    # write graph for tensorboard
    write = tf.summary.FileWriter(log_file, tf.get_default_graph())
    write.close()

    if os.path.isfile(log_file):
        os.remove(log_file)
        print("delete {}".format(log_file))

    if os.path.isfile(loss_file):
        os.remove(loss_file)
        print("delete {}".format(loss_file))

    round_number = int(TRAIN_FILE_LINE_NUM / TRAIN_BATCH_SIZE)
    for epoch_num in range(int(global_step.eval()) // round_number, EPOCH):
        accus = []
        test_accus = []
        for step in range(int(global_step.eval()) % round_number, round_number):

            input = sess.run(train_input_element)
            x1, x2, x3 = helper.convert_input(input, id2vector)

            _, _, loss_v, accu, _ = sess.run(
                [train_op, extra_update_ops, model.loss, model.accuracy, global_step], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3,
                    model.training: True})

            accus.append(accu)
            if step % 600 == 0:
                test_accu = sess.run([model.accuracy], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3,
                    model.training: False})

                test_accus.append(test_accu)

                print("epoch {}, step {}/{}, loss {}, accuracy {}, test accuracy {}, mean accu {}/{}"
                      .format(epoch_num, step, round_number, loss_v, accu, test_accu,
                              np.mean(accus), np.mean(test_accus)))

                saver.save(sess, model_save_path + model_name, global_step=global_step)

                helper.write_loss(loss_file, loss=loss_v)

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()

        saver.save(sess, model_save_path + model_name, global_step=global_step)
        # calculate accuracy on test data
        test_accuracy = []
        for test_step in range(TEST_FILE_LINE_NUM // TEST_BATCH_SIZE):
            test_input = sess.run(test_input_element)
            x1, x2, x3 = helper.convert_input(test_input, id2vector)
            accu = sess.run([model.accuracy], feed_dict={
                model.anchor_input: x1,
                model.positive_input: x2,
                model.negative_input: x3,
                model.training: False})
            test_accuracy.append(accu)
        print(" test accuracy {}".format(np.mean(test_accuracy)))


def make_dataset(file_name, buffer_size, batch_size, epoch):
    """

    Args:
        file_name:
        buffer_size:
        batch_size:
        epoch:

    Returns:

    """
    return tf.data.TextLineDataset(file_name) \
        .map(lambda s: tf.string_split([s], delimiter="\t").values) \
        .shuffle(buffer_size=buffer_size) \
        .batch(batch_size=batch_size).repeat(epoch)


def count_file_line(file_name):
    """

    Args:
        file_name:

    Returns:

    """
    p = os.popen('wc -l {}'.format(file_name))
    return int(p.read().strip().split(' ')[0])


def main(argv=None):
    """

    :param argv:
    :return:
    """

    print("************** start *****************")
    global TRAIN_FILE_LINE_NUM
    TRAIN_FILE_LINE_NUM = count_file_line(train_file_name)
    global TEST_FILE_LINE_NUM
    TEST_FILE_LINE_NUM = count_file_line(test_file_name)
    print("Train file has {} lines, test file has {} lines".format(TRAIN_FILE_LINE_NUM, TEST_FILE_LINE_NUM))
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
