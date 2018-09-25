# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/8/30 下午1:59
"""

import logging
import os
import sys
import traceback

import numpy as np
import tensorflow as tf

sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

import src.utils.tripletnetwork_helper as helper
from src.model.tripletnetwork_dssm_l1_long import TripletNetwork

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_BATCH_SIZE = 128
EPOCH = 5
TRAIN_BUFFER_SIZE = 8196
TRAIN_FILE_LINE_NUM = 0

TEST_BATCH_SIZE = 1024
TEST_FILE_LINE_NUM = 0

REGULARIZATION_RATE = 1e-4

MODE = "l1"
train_file_name = os.path.join(ROOT_PATH, 'data/simple/train_id.txt')
test_file_name = os.path.join(ROOT_PATH, 'data/simple/test_id.txt')
word2vec_file_name = os.path.join(ROOT_PATH, 'data/wordvec.vec')
model_save_path = os.path.join(ROOT_PATH, 'model_dssm_simple_{}/'.format(MODE))
model_name = "triplet_network.ckpt"
log_file = os.path.join(ROOT_PATH, 'log/triplet_network_dssm_{}'.format(MODE))
loss_file = os.path.join(ROOT_PATH, 'loss_dssm_{}/loss.txt'.format(MODE))


def train():
    """

    :return:
    """

    # make datasets
    train_data = make_dataset(train_file_name, TRAIN_BUFFER_SIZE, TRAIN_BATCH_SIZE, EPOCH)
    train_iterator = train_data.make_initializable_iterator()
    train_input_element = train_iterator.get_next()

    test_data = make_dataset(test_file_name, TEST_BATCH_SIZE, TEST_BATCH_SIZE, EPOCH)
    test_iterator = test_data.make_initializable_iterator()
    test_input_element = test_iterator.get_next()

    # build model
    word_vec = helper.get_word_vec(word2vec_file_name)
    regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZATION_RATE)
    model = TripletNetwork(25, word_vec, regularizer)

    # write graph for tensorboard
    logging.info([x.name for x in tf.global_variables()])
    write = tf.summary.FileWriter(log_file, tf.get_default_graph())
    write.close()

    logging.info("setting hyperparameters")
    global_step = tf.Variable(0.0, trainable=False, name="global_step")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # batch normalization need this op to update its variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(model.loss, global_step=global_step)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)

        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            logging.info("load model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.info("no existing model found")
            sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        if os.path.isfile(log_file):
            os.remove(log_file)
            logging.info("delete {}".format(log_file))

        if os.path.isfile(loss_file):
            os.remove(loss_file)
            logging.info("delete {}".format(loss_file))

        round_number = int(TRAIN_FILE_LINE_NUM / TRAIN_BATCH_SIZE)
        for epoch_num in range(int(global_step.eval()) // round_number, EPOCH):
            accus = []
            test_accus = []
            for step in range(int(global_step.eval()) % round_number, round_number):

                train_input = sess.run(train_input_element)
                x1, x2, x3 = helper.read_from_input(train_input)

                _, loss_v, accu, __ = sess.run(
                    [train_op, model.loss, model.accuracy, global_step], feed_dict={
                        model.anchor_input: x1,
                        model.positive_input: x2,
                        model.negative_input: x3,
                        model.training: True})

                accus.append(accu)
                if step % 1000 == 0:
                    test_accu, post_sim_l1, neg_sim_l1 = sess.run([model.accuracy, model.post_sim_l1, model.neg_sim_l1],
                                                                  feed_dict={
                                                                      model.anchor_input: x1,
                                                                      model.positive_input: x2,
                                                                      model.negative_input: x3,
                                                                      model.training: False})

                    if test_accu < 0.4:
                        logging.info(post_sim_l1)
                        logging.info(neg_sim_l1)

                    test_accus.append(test_accu)

                    logging.info("epoch {}, step {}/{}, loss {}, accuracy {}, test accuracy {}, mean accu {}/{}"
                                 .format(epoch_num, step, round_number, loss_v, accu, test_accu,
                                         np.mean(accus), np.mean(test_accus)))

                    saver.save(sess, model_save_path + model_name, global_step=global_step)

                    helper.write_loss(loss_file, loss=loss_v)

                if np.isnan(loss_v):
                    logging.info('Model diverged with loss = NaN')
                    quit()

            saver.save(sess, model_save_path + model_name, global_step=global_step)
            # calculate accuracy on test data
            test_accuracy = []
            for test_step in range(TEST_FILE_LINE_NUM // TEST_BATCH_SIZE):
                test_input = sess.run(test_input_element)
                x1, x2, x3 = helper.read_from_input(test_input)
                accu = sess.run([model.accuracy], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3,
                    model.training: False})
                test_accuracy.append(accu)
            logging.info(" test accuracy {}".format(np.mean(test_accuracy)))


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

    logging.info("************** start *****************")
    global TRAIN_FILE_LINE_NUM
    TRAIN_FILE_LINE_NUM = count_file_line(train_file_name)
    global TEST_FILE_LINE_NUM
    TEST_FILE_LINE_NUM = count_file_line(test_file_name)
    logging.info("Train file has {} lines, test file has {} lines".format(TRAIN_FILE_LINE_NUM, TEST_FILE_LINE_NUM))
    train()


if __name__ == '__main__':
    # sys.stdout = open(os.path.join(ROOT_PATH, "train_dssm.log"), "w")
    tf.logging.set_verbosity(tf.logging.INFO)
    logging.basicConfig(filename=os.path.join(ROOT_PATH, "train_dssm_simple.log"),
                        level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
                        filemode="w")
    try:
        tf.app.run()
    except Exception as e:
        print(traceback.format_exc())
        logging.error("{}:{}".format(traceback.format_exception(*sys.exc_info())[-2],
                                     traceback.format_exception(*sys.exc_info())[-1]))
        # logging.error("Unexpected exception! %s", e)