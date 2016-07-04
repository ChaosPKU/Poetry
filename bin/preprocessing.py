#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: preprocessing.py
# Date: 2016-06-01
# Author: Chaos <xinchaoxt@gmail.com>

import sys
sys.path.append('..')
from configs.config import *
from collections import Counter
import cPickle


def read_data():
    data = []
    for file in poem_raw_file_name:
        inf = open(poetry_raw_directory_path + file)
        fields = inf.readline().decode("utf-8").strip().split('\t')
        body_idx = fields.index('body')
        for line in inf:
            fields = line.decode('utf-8').strip().split('\t')
            # add start and end tags
            body = "^" + fields[body_idx] + "$"
            data.append([chr for chr in body if chr])
        inf.close()
    return data


def word2indices(words_list, words_to_indices, word):
    if word in words_list:
        return words_to_indices[word]
    else:
        return len(words_to_indices)


def preprocessing(raw_data, length, step, size):
    """
    Construct the training data
    :param length: length of the input of RNN
    :param step: timesteps of sampling
    :param size: the vocabulary size
    :return: dicts of words to indices, indices to words and training data and labels
    """
    data = []
    label = []

    words = Counter()
    for i in raw_data:
        words.update(i)
    vocabulary = words.most_common(size)
    words_list = [x[0] for x in vocabulary]
    words_to_indices = {x[0]: i + 1 for i, x in enumerate(vocabulary)}
    indices_to_words = {i + 1: x[0] for i, x in enumerate(vocabulary)}
    # use * to represent these words ignored in the vocabulary
    words_to_indices[u'*'] = size + 1
    indices_to_words[size + 1] = u'*'

    for line in raw_data:
        new_line = [line[max(i, 0): i + length] for i in xrange(1 - length, len(line) - length, step)]
        new_line = [[word2indices(words_list, words_to_indices, i) for i in x] for x in new_line]
        data.extend(new_line)
        label.extend([word2indices(words_list, words_to_indices, line[i + length]) for i in xrange(1 - length, len(line) - length, step)])

    return words_to_indices, indices_to_words, data, label


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    raw_data = read_data()
    words_to_indices, indices_to_words, data, label = preprocessing(raw_data, poem_max_length, poem_division_step,
                                                                    vocabulary_data_size)
    print "%d words used" % len(words_to_indices)
    print "%d train samples generated" % len(data)
    cPickle.dump([words_to_indices, indices_to_words], open(words_indices_dict_path, 'w+'))
    nb = len(data) / nb_train_samples
    for i in xrange(nb):
        cPickle.dump([data[(i * nb_train_samples):(i + 1) * nb_train_samples],
                      label[(i * nb_train_samples):(i + 1) * nb_train_samples]],
                     open(poetry_train_data_path[:-4] + '_' + (('%0' + str(len(str(nb))) + 'd') % i) + poetry_train_data_path[-4:], 'w+'))
    cPickle.dump([data[nb * nb_train_samples:len(data)], label[nb * nb_train_samples:len(data)]],
                 open(poetry_train_data_path[:-4] + '_' + (('%0' + str(len(str(nb))) + 'd') % nb) + poetry_train_data_path[-4:], 'w+'))
