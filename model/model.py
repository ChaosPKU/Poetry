#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: model.py
# Date: 2016-06-03
# Author: Chaos <xinchaoxt@gmail.com>

import numpy as np
from configs.config import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


class LSTM_RNN_Model:
    def __init__(self, X, Y, X_test, Y_test, input_len=32, hidden_len=512, output_len=100, dropout=0.2, nb_epoch=100, batch_size=256, model_architecture_file=model_architecture_file_path, model_weights_file=model_weights_path):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.dropout = dropout
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.model_architecture_file = model_architecture_file
        self.model_weights_file = model_weights_file
        self.model = Sequential()

    def build(self):
        self.model.add(Embedding(self.output_len + 1, self.output_len + 1, weights=[np.identity(self.output_len + 1)],
                                 input_length=self.input_len, trainable=False))
        self.model.add(LSTM(self.hidden_len, input_shape=(self.input_len, self.output_len + 1), return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(LSTM(self.hidden_len, return_sequences=False))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.output_len))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

    def train(self):
        early_stop = EarlyStopping(verbose=1, patience=3, monitor='val_loss')
        model_check = ModelCheckpoint(self.model_architecture_file, monitor='val_loss', verbose=True, save_best_only=True)
        self.model.fit(self.X, self.Y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                       validation_data=(self.X_test, self.Y_test), callbacks=[early_stop, model_check])
        outf = open(self.model_architecture_file, 'w')
        outf.write(self.model.to_json())
        self.model.save_weights(self.model_weights_file)
        outf.close()

