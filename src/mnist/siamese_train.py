# -*- coding: utf-8 -*-
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.
By Youngwook Paul Kwon (young at berkeley.edu)
https://github.com/ywpkwon/siamese_tf_mnist
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from builtins import input

import numpy as np
import tensorflow as tf
# import system things
from tensorflow.examples.tutorials.mnist import input_data  # for data

# import helpers
import siamese_inference as inference
import siamese_visualize as visualize

# prepare data and tf.session
mnist = input_data.read_data_sets("../data/mnist_data", one_hot=False)
sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
load = False
model_ckpt = './model/siamese.meta'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True

# start training
if load:
    saver.restore(sess, './model/siamese')

for step in range(50000):
    batch_x1, batch_y1 = mnist.train.next_batch(128)
    batch_x2, batch_y2 = mnist.train.next_batch(128)
    batch_y = (batch_y1 == batch_y2).astype('float')

    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
        siamese.x1: batch_x1,
        siamese.x2: batch_x2,
        siamese.y_: batch_y})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if step % 10 == 0:
        print('step %d: loss %.3f' % (step, loss_v))

    if step % 100 == 0 and step > 0:
        saver.save(sess, './model/siamese')
        # 相当于对于test中10000个数据的每个数据，用当前模型转成一个二维向量，后面画图可以看出，相同图片向量基本在一起，有明显聚类特性
        embed = siamese.o1.eval({siamese.x1: mnist.test.images})
        embed.tofile('./model/embed.txt')

# visualize result
x_test = mnist.test.images.reshape([-1, 28, 28])
y_test = mnist.test.labels
visualize.visualize(embed, x_test, y_test)
