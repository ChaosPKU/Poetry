#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: preprocessing.py
# Date: 2016-06-24
# Author: Chaos <xinchaoxt@gmail.com>

import sys
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
            body = "^" + fields[body_idx] + "$"
            data.append([chr for chr in body if chr])
        inf.close()
    return data


def preprocessing(raw_data, length, step):
    data = []
    label = []

    words = Counter()
    for i in raw_data:
        words.update(i)
    words_to_indices = {x[0]: i + 1 for i, x in enumerate(words.most_common())}
    indices_to_words = {i + 1: x[0] for i, x in enumerate(words.most_common())}

    for line in raw_data:
        new_line = [line[max(i, 0): i + length] for i in xrange(1 - length, len(line) - length, step)]
        new_line = [[words_to_indices[i] for i in x] for x in new_line]
        data.extend(new_line)
        label.extend([words_to_indices[line[i + length]] for i in xrange(1 - length, len(line) - length, step)])

    return words_to_indices, indices_to_words, data, label


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    raw_data = read_data()
    words_to_indices, indices_to_words, data, label = preprocessing(raw_data, poem_max_length, poem_division_step)
    print "%d words found" % len(words_to_indices)
    print "%d train samples generated" % len(data)
    cPickle.dump([words_to_indices, indices_to_words], open(words_indices_dict_path, 'w+'))
    cPickle.dump([data, label], open(poetry_train_data_path, 'w+'))

    # 11780 words found
    # 14688545 train samples generated
