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

