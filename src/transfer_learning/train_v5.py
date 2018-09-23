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
from src.model.tripletnetwork_v5 import TripletNetwork

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 128
EPOCH = 5
BUFFER_SIZE = 8196
FILE_LINE_NUM = 0
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

train_file_name = os.path.join(ROOT_PATH, 'data/train_tokenize.txt')
test_file_name = os.path.join(ROOT_PATH, 'data/test_tokenize.txt')
word2vec_file_name = os.path.join(ROOT_PATH, 'data/model.vec')
model_save_path = os.path.join(ROOT_PATH, 'model_v5/')
model_name = "triplet_network.ckpt"
log_file = os.path.join(ROOT_PATH, 'log_v5/triplet_network')
loss_file = os.path.join(ROOT_PATH, 'loss_v5/loss.txt')

bottlenect_layer_training = False


def train():
    """

    :param train_data:
    :return:
    """
    train_data = make_dataset(train_file_name)
    iterator = train_data.make_initializable_iterator()
    input_element = iterator.get_next()

    id2vector = helper.get_id_vector()

    model = TripletNetwork(25, 256, trainable=bottlenect_layer_training)

    global_step = tf.Variable(0.0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001,
                                               global_step=global_step,
                                               decay_steps=10000, decay_rate=0.9,
                                               staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(model.loss, global_step=global_step)

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
        print("load {} successfully".format(ckpt.model_checkpoint_path))
    else:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    print([x.name for x in tf.global_variables()])

    write = tf.summary.FileWriter(log_file, tf.get_default_graph())
    write.close()

    if os.path.isfile(log_file):
        os.remove(log_file)
        print("delete {}".format(log_file))

    round_number = int(FILE_LINE_NUM / BATCH_SIZE)
    if not bottlenect_layer_training:
        current_global_steps = 0
    else:
        current_global_steps = int(global_step.eval())

    for epoch_num in range(current_global_steps // round_number, EPOCH):
        accus = []
        test_accus = []
        for step in range(current_global_steps % round_number, round_number):

            input = sess.run(input_element)
            x1, x2, x3 = helper.get_input_embedding(input, id2vector)

            _, loss_v, accu, __ = sess.run(
                [train_op, model.loss, model.accuracy, global_step], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3,
                    model.training: True})

            accus.append(accu)
            if step % 600 == 0:
                test_accu2 = sess.run([model.accuracy], feed_dict={
                    model.anchor_input: x1,
                    model.positive_input: x2,
                    model.negative_input: x3,
                    model.training: True})

                test_accus.append(test_accu2)

                print("epoch {}, step {}/{}, loss {}, accuracy {}, test accuracy {}, "
                      "mean accu {}/{}".format(epoch_num, step, round_number, loss_v,
                                               accu, test_accu2, np.mean(accus),
                                               np.mean(test_accus)))
                saver.save(sess, model_save_path + model_name, global_step=global_step)

                helper.write_loss(loss_file, loss=loss_v)

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
