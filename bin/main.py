#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: main.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>
# from keras.models import Sequential
# from keras.layers import Dense
# from seq2seq.models import AttentionSeq2seq, SimpleSeq2seq
from lib import *
# from model.model import get_GRU_model


if __name__ == '__main__':
    # X, Y = load_data(GRU_model_nb_samples,
    #                  GRU_model_input_nb_words, GRU_model_output_nb_words)
    # model = get_GRU_model()
    # print 'Training...'
    # model.fit(X, Y,
    #           nb_epoch=GRU_model_nb_epoch,
    #           batch_size=32,
    #           verbose=1)
    # model.save_weights(model_weights_path, overwrite=True)
    lstm_train()
