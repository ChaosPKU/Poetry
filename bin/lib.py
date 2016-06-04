#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: lib.py
# Date: 2016-06-04
# Author: Chaos <xinchaoxt@gmail.com>

from configs.config import *
from gensim.models import Word2Vec
import numpy as np
import re

word2vec_model = Word2Vec.load(poetry_gen_data_model_path)
vocab_size = len(word2vec_model.vocab)


def get_word_vector(word):
    if word in word2vec_model.vocab.keys():
        return word2vec_model[word]
    else:
        return [0] * word_vector_dimension


def load_data(nb_samples, X_word, Y_word):
    X = []
    Y = []
    inf = open(poetry_vocabulary_data_path, 'r')
    cnt = 0
    for line in inf:
        cnt += 1
        sens = re.split(' |\t', line.decode('utf-8').strip())
        X.append(np.array([get_word_vector(x) for x in sens[:X_word]]).flatten())
        Y.append(np.array([get_word_vector(x) for x in sens[:Y_word]]).flatten())
        if cnt == nb_samples:
            break
    return np.array(X), np.array(Y)


def get_vocabulary_list():
    return word2vec_model.vocab.keys()


