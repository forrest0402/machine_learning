# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import sys
import numpy as np
import math
from src.model.lstm import LSTMLM

"""
@Author: xiezizhe 
@Date: 2018/11/15 5:15 PM
"""
sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                 os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN = True


def gen_vocab(file_path, file_name):
    if os.path.exists(file_path + "vocab"):
        return len(open(file_path + "vocab", 'r').readlines())

    word_list = list(
            {t for line in open(file_path + file_name, 'r') for t in line.strip().replace("<unk>", "_UNK_").split()})
    word_list = ["_PAD_", "_BOS_", "_EOS_"] + word_list

    print(len(word_list))

    with open(file_path + "vocab", "w") as vocab_file:
        for word in word_list:
            vocab_file.write(word + "\n")

    return len(word_list)


def gen_id_seqs(file_path, file_name):
    if os.path.exists(file_path + file_name.strip('.txt') + ".ids"):
        return len(open(file_path + file_name.strip('.txt') + ".ids", 'r').readlines())

    def word_to_id(word, word_dict):
        id = word_dict.get(word)
        return id if id is not None else word_dict.get("_UNK_")

    with open(file_path + "vocab", "r") as vocab_file:
        lines = [line.strip() for line in vocab_file.readlines()]
        word_dict = dict([(b, a) for (a, b) in enumerate(lines)])

    with open(file_path + file_name, "r") as raw_file:
        with open(file_path + file_name.strip('.txt') + ".ids", "w") as current_file:
            for line in raw_file.readlines():
                line = [word_to_id(word, word_dict) for word in line.strip().replace("<unk>", "_UNK_").split()]
                # each sentence has the start and the end.
                line_word_ids = [1] + line + [2]
                current_file.write(" ".join([str(id) for id in line_word_ids]) + "\n")


def create_model(sess):
    model = LSTMLM(batch_size=128,
                   vocab_size=vocab_size,
                   num_hidden_units=100,
                   dropout_rate=1.0)
    sess.run(tf.global_variables_initializer())
    return model


def batch_train(sess, model, saver, num_epochs, num_train_samples, learning_rate, max_gradient_norm,
                check_point_step):
    best_score = np.inf
    patience = 5
    epoch = 0
    params = tf.trainable_variables()
    global_step = tf.Variable(0, trainable=False)
    opt = tf.train.AdagradOptimizer(learning_rate)
    gradients = tf.gradients(model.loss, params, colocate_gradients_with_ops=True)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    while epoch < num_epochs:
        sess.run(model.trining_init_op, {model.file_name_train: ROOT_PATH + "/data/ptb.train.ids"})
        train_loss = 0.0
        train_valid_words = 0
        while True:
            try:
                _loss, _valid_words, global_step, current_learning_rate, _ = sess.run(
                        [model.loss, model.valid_words, global_step, learning_rate, updates],
                        {model.dropout_rate: 0.5})
                train_loss += np.sum(_loss)
                train_valid_words += _valid_words

                if global_step % check_point_step == 0:
                    train_loss /= train_valid_words
                    train_ppl = math.exp(train_loss)
                    print("Training Step: {}, LR: {}".format(global_step, current_learning_rate))
                    print("Training PPL: {}".format(train_ppl))

                    train_loss = 0.0
                    train_valid_words = 0


            except tf.errors.OutOfRangeError:
                # The end of one epoch
                break

        sess.run(model.validation_init_op, {model.file_name_validation: "./data/valid.ids"})
        dev_loss = 0.0
        dev_valid_words = 0
        while True:
            try:
                _dev_loss, _dev_valid_words = sess.run(
                        [model.loss, model.valid_words],
                        {model.dropout_rate: 1.0})

                dev_loss += np.sum(_dev_loss)
                dev_valid_words += _dev_valid_words

            except tf.errors.OutOfRangeError:
                dev_loss /= dev_valid_words
                dev_ppl = math.exp(dev_loss)
                print("Validation PPL: {}".format(dev_ppl))
                if dev_ppl < best_score:
                    patience = 5
                    saver.save(sess, "model/best_model.ckpt")
                    best_score = dev_ppl
                else:
                    patience -= 1

                if patience == 0:
                    epoch = model.num_epochs

                break


if __name__ == '__main__':
    print("start")
    vocab_size = gen_vocab(ROOT_PATH + '/data/', 'ptb.train.txt')
    num_train_samples = gen_id_seqs(ROOT_PATH + '/data/', 'ptb.train.txt')
    num_valid_samples = gen_id_seqs(ROOT_PATH + '/data/', 'ptb.valid.txt')

    if TRAIN:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = create_model(sess)
            saver = tf.train.Saver()
            batch_train(sess, model, saver,
                        num_epochs=80,
                        learning_rate=0.01,
                        max_gradient_norm=5.0,
                        num_train_samples=num_train_samples,
                        check_point_step=100)
    else:
        tf.reset_default_graph()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #     model = create_model(sess)
        #     saver = tf.train.Saver()
        #     saver.restore(sess, "model/best_model.ckpt")
        #     predict_id_file = os.path.join("data", test_file.split("/")[-1] + ".ids")
        #     if not os.path.isfile(predict_id_file):
        #         gen_id_seqs(test_file)
        #     predict(sess, predict_id_file, test_file, verbose=True)
    print("hello, world")
