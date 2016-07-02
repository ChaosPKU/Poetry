#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: lib.py
# Date: 2016-06-04
# Author: Chaos <xinchaoxt@gmail.com>

import sys
sys.path.append('..')
from configs.config import *
import numpy as np


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

