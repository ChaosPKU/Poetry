#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: lib.py
# Date: 2016-06-04
# Author: Chaos <xinchaoxt@gmail.com>

import sys
sys.path.append('..')
from configs.config import *
import numpy as np
import os
import cPickle
from model.model import LSTM_RNN_Model
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def make_poem(model, start_words, words_to_indices, indices_to_words):
    text = [0] * poem_max_length
    start_words = '^' + start_words
    start_words = start_words.decode('utf-8')
    start_indices = [words_to_indices[i] for i in start_words]

    text.extend(start_indices)

    for i in xrange(4 * poem_max_length):
        inputs = np.asarray([text[-poem_max_length:]])
        cnt = 0
        while cnt < 10:
            cnt += 1
            preds = model.predict(inputs, verbose=0)[0]
            next_index = sample(preds, sample_temperature) + 1
            if next_index != words_to_indices['*'.decode('utf-8')]:
                break
        text.append(next_index)
        if next_index == words_to_indices['$'.decode('utf-8')]:
            break

    # for i in xrange(rows):
    #     j = 0
    #     while true:
    #         inputs = np.asarray([text[-poem_max_length:]])
    #         preds = model.predict(inputs, verbose=0)[0]
    #         next_index = sample(preds, sample_temperature) + 1
    #         if indices_to_words[next_index].encode("utf-8") in (puncts + tags):

    return [indices_to_words[i].encode("utf-8") for i in text[poem_max_length + 1:-1]]


def get_model(args, words_to_indices):
    model = None
    if not os.path.exists(args.model) or not os.path.exists(args.weight):
        print 'No available model exists. Start Training...'
        all_train_files = os.listdir(poetry_train_data_dir)
        train_file_list = [all_train_files[i * args.files:min(len(all_train_files), (i + 1) * args.files)]
                           for i in xrange((len(all_train_files) + args.files - 1) / args.files)]
        for l in xrange(len(train_file_list)):
            print 'Training process %d' % l
            data = []
            labels = []
            for i in xrange(args.files):
                data_file_name = poetry_train_data_path[:-4] + '_' + \
                                 ('%0' + str(len(str(len(all_train_files)))) + 'd') % (i + l * args.files) + \
                                 poetry_train_data_path[-4:]
                print 'Read data from file ' + data_file_name
                new_data, new_labels = cPickle.load(open(data_file_name, 'r+'))
                data.extend(new_data)
                labels.extend(new_labels)
            num_words = len(words_to_indices)
            data = sequence.pad_sequences(data, maxlen=poem_max_length)
            labels = to_categorical([i - 1 for i in labels], num_words)
            print 'Data shape: ', data.shape
            train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1)
            if not os.path.exists(args.model) or not os.path.exists(args.weight):
                print 'Building...'
                lstm_model = LSTM_RNN_Model(train_data, train_labels, test_data, test_labels, output_len=num_words,
                                            nb_epoch=args.epoch, batch_size=args.batch)
                lstm_model.build()
            else:
                print 'Loading model...'
                pre_model = model_from_json(open(args.model).read())
                pre_model.load_weights(args.weight)
                lstm_model = LSTM_RNN_Model(train_data, train_labels, test_data, test_labels, output_len=num_words,
                                            nb_epoch=args.epoch, batch_size=args.batch, model=pre_model)
            print 'Training...'
            lstm_model.train()
            model = lstm_model.model
    else:
        print 'Loading model...'
        model = model_from_json(open(args.model).read())
        model.load_weights(args.weight)
    return model

