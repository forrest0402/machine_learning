# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/8/30 下午1:59
"""

import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

import src.model.model_name as ModelName
import src.utils.tripletnetwork_helper as helper
from src.model.tripletnetwork_mode import ModelManager

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_BATCH_SIZE = 128
EPOCH = 3
TRAIN_BUFFER_SIZE = 8196
TRAIN_FILE_LINE_NUM = 0

TEST_BATCH_SIZE = 128
TEST_FILE_LINE_NUM = 0

REGULARIZATION_RATE = 1e-4

TRAIN_FILE = os.path.join(ROOT_PATH, 'data/train_id.txt')
TEST_FILE = os.path.join(ROOT_PATH, 'data/test_id.txt')
WORD_EMBEDDING_FILE = os.path.join(ROOT_PATH, 'data/wordvec.vec')
SAVE_MODEL_NAME = "triplet_network.ckpt"
EXE_MODEL_NAME = ModelName.ResCNN


def test(model, sess, test_input_element, word_vec):
    test_accuracy1, test_accuracy2 = [], []
    for test_step in range(TEST_FILE_LINE_NUM // TEST_BATCH_SIZE):
        test_input = sess.run(test_input_element)
        x1, x2, x3 = helper.read_from_input(test_input, word_vec)
        accu = sess.run([model.accuracy], feed_dict={
            model.anchor_input  : x1,
            model.positive_input: x2,
            model.negative_input: x3,
            model.training      : False})
        test_accuracy1.append(accu)
        accu = sess.run([model.accuracy], feed_dict={
            model.anchor_input  : x2,
            model.positive_input: x1,
            model.negative_input: x3,
            model.training      : False})
        test_accuracy2.append(accu)
    logging.info(" test accuracy {}/{}".format(np.mean(test_accuracy1), np.mean(test_accuracy2)))


def train(argv=None):
    """

    :return:
    """
    logging.info("making datasets")
    train_data = make_dataset(TRAIN_FILE, TRAIN_BUFFER_SIZE, TRAIN_BATCH_SIZE, EPOCH)
    train_iterator = train_data.make_initializable_iterator()
    train_input_element = train_iterator.get_next()

    test_data = make_dataset(TEST_FILE, TEST_BATCH_SIZE, TEST_BATCH_SIZE, 1000000)
    test_iterator = test_data.make_initializable_iterator()
    test_input_element = test_iterator.get_next()

    logging.info("initializing mode")
    mode = ModelManager()

    logging.info("creating the model")
    model, suffix, word_vec = mode.get(FLAGS.suffix)
    model_save_path = os.path.join(ROOT_PATH, 'model_{}/'.format(suffix))
    log_file = os.path.join(ROOT_PATH, 'log/{}'.format(suffix))
    loss_file = os.path.join(ROOT_PATH, 'loss/loss_{}.txt'.format(suffix))

    # write graph for tensorboard
    logging.info([x.name for x in tf.global_variables()])

    logging.info("setting hyperparameters")
    global_step = tf.Variable(0.0, trainable=False, name="global_step")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # batch normalization need this op to update its variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(model.cal_loss, global_step=global_step)

    # delete files
    if os.path.exists(log_file):
        os.system("rm -rf {}".format(log_file))
        logging.info("delete {}".format(log_file))

    if os.path.isfile(loss_file):
        os.remove(loss_file)
        logging.info("delete {}".format(loss_file))

    if FLAGS.rerun and os.path.exists(model_save_path):
        os.system("rm -rf {}".format(model_save_path))
        logging.info("delete {}".format(model_save_path))

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
            # test(model, sess, test_input_element, word_vec)
        else:
            logging.info("no existing model found")
            sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        merge = tf.summary.merge_all()
        write = tf.summary.FileWriter(log_file, tf.get_default_graph())

        logging.info("start to train")
        round_number = int(TRAIN_FILE_LINE_NUM / TRAIN_BATCH_SIZE)
        for epoch_num in range(int(global_step.eval()) // round_number, EPOCH):
            accus = []
            test_accus = []
            for step in range(int(global_step.eval()) % round_number, round_number):

                train_input = sess.run(train_input_element)
                x1, x2, x3 = helper.read_from_input(train_input, word_vec)

                summary, _, loss_v, accu, __ = sess.run(
                        [merge, train_op, model.cal_loss, model.accuracy, global_step], feed_dict={
                            model.anchor_input  : x1,
                            model.positive_input: x2,
                            model.negative_input: x3,
                            model.training      : True})
                write.add_summary(summary, global_step=global_step.eval())
                accus.append(accu)
                if step % 10000 == 0 or step == 1000 or step == 2000:
                    test(model, sess, test_input_element, word_vec)
                if step % 1000 == 0:
                    # test
                    # gamma = tf.get_default_graph().get_tensor_by_name("triplet/output_bn/gamma:0")
                    # beta = tf.get_default_graph().get_tensor_by_name("triplet/output_bn/beta:0")
                    # mean = tf.get_default_graph().get_tensor_by_name("triplet/output_bn/moving_mean:0")
                    # variance = tf.get_default_graph().get_tensor_by_name("triplet/output_bn/moving_variance:0")
                    # print("mean = {}, variance = {}".format(sess.run(mean), sess.run(variance)))

                    test_accu, pos_sim, neg_sim = sess.run([model.accuracy, model.pos_sim, model.neg_sim],
                                                           feed_dict={
                                                               model.anchor_input  : x1,
                                                               model.positive_input: x2,
                                                               model.negative_input: x3,
                                                               model.training      : False})

                    if test_accu < 0.3:
                        logging.info(np.average(pos_sim))
                        logging.info(np.average(neg_sim))

                    test_accus.append(test_accu)

                    logging.info(
                            "epoch {}, step {}/{}, loss {}, accu {}, test accu {}, pos/neg sim: {}/{}, mean accu {}/{}"
                                .format(epoch_num, step, round_number, loss_v, accu, test_accu, np.average(pos_sim),
                                        np.average(neg_sim), np.mean(accus), np.mean(test_accus)))

                    saver.save(sess, model_save_path + SAVE_MODEL_NAME, global_step=global_step)
                    helper.write_loss(loss_file, loss=loss_v)

                if np.isnan(loss_v):
                    logging.info('Model diverged with loss = NaN')
                    quit()

            # calculate accuracy on test data
            test(model, sess, test_input_element, word_vec)
            try:
                constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                              ['cosine_similarity/div'])
                with tf.gfile.FastGFile(model_save_path + 'triplet_{}_{}.pb'.format(suffix, epoch_num),
                                        mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
            except Exception as e:
                print(e.__cause__)

        write.close()


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
    if FLAGS.rerun:
        logging.info("************** start *****************")
        logging.info("tf version: {}".format(tf.__version__))
        logging.info("python version: {}".format(sys.version.split(" ")[0]))
        logging.info(FLAGS)
    else:
        logging.info("************** restart *****************")

    global TRAIN_FILE_LINE_NUM
    TRAIN_FILE_LINE_NUM = 22208057  # count_file_line(TRAIN_FILE)
    global TEST_FILE_LINE_NUM
    TEST_FILE_LINE_NUM = 344079  # count_file_line(TEST_FILE)
    logging.info("Train file has {} lines, test file has {} lines".format(TRAIN_FILE_LINE_NUM, TEST_FILE_LINE_NUM))
    train(argv)
    logging.info("Hello, world")


if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('true', 't', '1'):
            return True
        elif v.lower() in ('false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expedted.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", type=str2bool, default=True)
    parser.add_argument("--suffix", type=str, default=EXE_MODEL_NAME)
    FLAGS, unparsed = parser.parse_known_args()
    # sys.stdout = open(os.path.join(ROOT_PATH, "train_dssm.log"), "w")
    tf.logging.set_verbosity(tf.logging.ERROR)
    if FLAGS.rerun:
        filemode = 'w'
    else:
        filemode = 'a'
    logging.basicConfig(filename=os.path.join(ROOT_PATH, "train_{}.log".format(FLAGS.suffix)),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
                        filemode=filemode)

    tf.app.run()
