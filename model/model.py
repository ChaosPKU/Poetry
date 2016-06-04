#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: model.py
# Date: 2016-06-03
# Author: Chaos <xinchaoxt@gmail.com>

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import random
# import sys
from configs.config import *


class LSTM_RNN_Model:
    def __init__(self, input_len, hidden_len, output_len, return_sequence=True):
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.return_sequence = return_sequence
        self.model = Sequential()

    def build(self, dropout=0.2):
        self.model.add(LSTM(input_dim=self.input_len, output_dim=self.hidden_len, return_sequences=self.return_sequence))
        self.model.add(Dropout(dropout))
        self.model.add(LSTM(input_dim=self.hidden_len, output_dim=self.hidden_len, return_sequences=False))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(input_dim=self.hidden_len, output_dim=self.output_len))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def sample(self, a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))


def get_data():
    inf = open(poetry_vocabulary_data_path, 'r')
    train_text = []
    for line in inf:
        for sentence in line.split('\t'):
            train_text += sentence.split()
        if len(train_text) > model_nb_samples:
            break
    train_text = train_text[:model_nb_samples + 1]
    words = set(train_text)
    word_to_indices = dict((word, idx) for idx, word in enumerate(words))
    indices_to_word = dict((idx, word) for idx, word in enumerate(words))
    X = np.zeros((model_nb_samples, input_max_len, len(words)), dtype=np.float32)
    Y = np.zeros((model_nb_samples, len(words)), dtype=np.float32)
    for i, word in enumerate(train_text[:-1]):
        X[i, 0, word_to_indices[word]] = 1
        Y[i, word_to_indices[train_text[i + 1]]] = 1
    return X, Y, word_to_indices, indices_to_word, len(words), train_text


def train():
    X, Y, word_to_indices, indices_to_word, nb_words, train_text = get_data()
    lstm = LSTM_RNN_Model(nb_words, 1000, nb_words)
    lstm.build(dropout=0.2)
    print "Vocabulary Length: %d, Text Length: %d" % (nb_words, len(train_text))
    for iter in xrange(model_nb_epoch):
        print("==============================================================")
        print("Iteration: ", iter)
        lstm.model.fit(X, Y, batch_size=model_batch_size, nb_epoch=1)
        start_index = random.randint(0, len(train_text) - 1)
        result = [train_text[start_index]]
        for i in xrange(27):
            seed = np.zeros((1, 1, nb_words))
            seed[0, 0, word_to_indices[result[-1]]] = 1
            predictions = lstm.model.predict(seed, verbose=0)[0]
            next_index = lstm.sample(predictions)
            next_word = indices_to_word[next_index]
            result.append(next_word)
        print ''.join(result)


if __name__ == '__main__':
    train()
