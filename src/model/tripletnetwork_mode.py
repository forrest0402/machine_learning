# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from src.model.plain_cnn import PlainCNN as CNN
from src.model.tripletnetwork_cosine import TextCnn as TextCnnCosine
from src.model.res_cnn import ResCNN
import src.utils.tripletnetwork_helper as helper
import src.model.model_name as ModelName
from src.model.hierarchical_cnn import HierarchicalCNN as HCnn

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

REGULARIZATION_RATE = 1e-4
word2vec_file_name = os.path.join(ROOT_PATH, 'data/wordvec.vec')


class ModelManager:

    def __init__(self):
        self.word_vec = helper.get_word_vec(word2vec_file_name)

    def cnn(self):
        model = CNN(32, 128)
        suffix = "plainCnn"
        return model, suffix, self.word_vec

    def text_cnn(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZATION_RATE)
        model = TextCnnCosine(32, 128, regularizer=regularizer)
        suffix = "textCnnCosine"
        return model, suffix, self.word_vec

    def resnet_cnn(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZATION_RATE)
        model = ResCNN(32, 128, regularizer=regularizer)
        suffix = "ResCNN"
        return model, suffix, self.word_vec

    def hcnn(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZATION_RATE)
        model = HCnn(32, 128, regularizer=regularizer)
        suffix = "hcnn"
        return model, suffix, self.word_vec

    def get(self, name):
        if name == ModelName.CNN:
            return self.cnn()
        elif name == ModelName.TextCnnCosine:
            return self.text_cnn()
        elif name == ModelName.ResCNN:
            return self.resnet_cnn()
        elif name == ModelName.HCnn:
            return self.hcnn()
        else:
            raise Exception("unable to find {}".format(name))
