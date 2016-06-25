#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: main.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>

# from keras.models import Sequential
# from keras.layers import Dense
# from seq2seq.models import AttentionSeq2seq, SimpleSeq2seq
from lib import *
from model.model import LSTM_RNN_Model
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
import argparse
import os
import cPickle


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
    # lstm_train()
    parser = argparse.ArgumentParser(description='Generate new Chinese poems')
    parser.add_argument('start_words', type=str, help="Chinese characters in the beginning of the poem")
    parser.add_argument('-m', '--model', default=model_architecture_file_path, help='Model architecture file location')
    parser.add_argument('-w', '--weight', default=model_weights_path, help='Model weights location')
    parser.add_argument('-s', '--samples', default=nb_train_samples, help='Number of train samples')
    parser.add_argument('-e', '--epoch', default=nb_train_epoch, help='Number of train epoch')

    args = parser.parse_args()

    words_to_indices, indices_to_words = cPickle.load(open(words_indices_dict_path, 'r+'))

    if not os.path.exists(args.model) or not os.path.exists(args.weight):
        print 'No available model exists.Start Training...'
        data, labels = cPickle.load(open(poetry_train_data_path, 'r+'))
        # data, labels = cPickle.load(open("../data/gen_data/small_train_data.pkl", 'r+'))
        # cPickle.dump([data[:1000], labels[:1000]], open("../data/gen_data/small_train_data.pkl", 'w'))
        num_words = len(words_to_indices)
        data = sequence.pad_sequences(data, maxlen=poem_max_length)
        labels = to_categorical([i - 1 for i in labels], num_words)
        print 'Data shape: ', data.shape
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
        model = LSTM_RNN_Model(train_data, train_labels, test_data, test_labels, output_len=num_words, nb_epoch=args.epoch, batch_size=32)
        model.build()
        print 'Training...'
        model.train()

    model = model_from_json(open(args.model).read())
    model.load_weights(args.weight)
    outf = open(console_log_path, 'a')
    for i in xrange(1000):
        poem = make_poem(model, args.start_words[1:-1], words_to_indices, indices_to_words)
        outf.write(''.join(poem) + '\n')
    outf.close()
