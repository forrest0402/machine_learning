# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from src.mnist import cnn_inference
from src.mnist import cnn_train


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           [5000, cnn_inference.IMAGE_SIZE, cnn_inference.IMAGE_SIZE, cnn_inference.NUM_CHANNELS],
                           name="x-input")
        y_ = tf.placeholder(tf.float32, [None, cnn_inference.OUTPUT_NODE], name="y-input")
        reshaped_images = np.reshape(mnist.validation.images, [5000, cnn_inference.IMAGE_SIZE, cnn_inference.IMAGE_SIZE,
                                                               cnn_inference.NUM_CHANNELS])
        validate_feed = {x: reshaped_images, y_: mnist.validation.labels}

        y = cnn_inference.inference(x, False, None)
        correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(cnn_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(cnn_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After {} training steps, accuracy is {}".format(global_step, accuracy_score))
            else:
                print("No checkpoint file found")


def main(argv=None):
    mnist = input_data.read_data_sets("../data/mnist_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
