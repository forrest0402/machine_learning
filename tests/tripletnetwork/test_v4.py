# -*- coding: utf-8 -*-
"""
@Author: xiezizhe
@Date: 2018/9/13 下午4:59
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.extend([os.path.dirname(os.path.dirname(__file__)), os.path.dirname(__file__)])

import src.utils.tripletnetwork_helper as helper
from src.model.tripletnetwork_v4 import TripletNetwork
import train_v4 as train

BATCH_SIZE = 1024


def make_dataset(file_name):
    """
    :param file_name:
    :return:
    """
    return tf.data.TextLineDataset(file_name) \
        .map(lambda s: tf.string_split([s], delimiter="\t").values) \
        .batch(batch_size=BATCH_SIZE)


if __name__ == '__main__':
    with tf.Graph().as_default() as g:
        print("************************** tests *******************************")

        p = os.popen('wc -l {}'.format(train.test_file_name))
        file_line_num = int(p.read().strip().split(' ')[0])
        print("Test file has {} lines".format(file_line_num))
        model = TripletNetwork(25, 256)
        # read test file
        print("read test file")

        test_data = make_dataset(train.test_file_name)
        iterator = test_data.make_initializable_iterator()
        input_element = iterator.get_next()

        # read word embedding
        id2vector = helper.get_id_vector()

        # print("start to construct inputs")
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        with tf.Session(config=tf_config) as sess:

            print("start to calculate accuracy")
            sess.run(iterator.initializer)

            ckpt = tf.train.get_checkpoint_state(train.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                helper.write_to_log(train.log_file, tf.summary.FileWriter, tf.get_default_graph())
                # write = tf.summary.FileWriter(train.log_file, tf.get_default_graph())
                # write.close()

                # builder = tf.saved_model.builder.SavedModelBuilder("model/triplet_network")
                # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
                # builder.save()

                accus = []
                try:
                    step = 0
                    while True:
                        input = sess.run(input_element)
                        x1, x2, x3 = helper.convert_input(input, id2vector)
                        accu = sess.run([model.accuracy],
                                        feed_dict={
                                            model.anchor_input: x1,
                                            model.positive_input: x2,
                                            model.negative_input: x3,
                                            model.training: False})
                        accus.append(accu)
                        print("{}/{},\taccu {}, average accuracy: {}".format(step * BATCH_SIZE,
                                                                             file_line_num, accu,
                                                                             np.mean(accus)))
                        step += 1
                except tf.errors.OutOfRangeError:
                    print("end!")
                    print("test accuracy: {}".format(np.mean(accus)))

            else:
                print("no saved model found")
