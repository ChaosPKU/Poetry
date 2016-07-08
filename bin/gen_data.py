# -*- encoding: utf-8 -*-

import re
from config import *

from collections import Counter
import cPickle


def read_raw_data():
    data = []
    for filename in RAW_DATA_FILES:
        inf = open(RAW_DATA_DIR + filename)
        fields = inf.readline().decode("utf-8").strip().split('\t')
        body_idx = fields.index(u'body')
        for line in inf:
            fields = line.decode('utf-8').strip().split('\t')
            data.append(fields[body_idx])
        inf.close()
    return data


def make_word_indices(data):
    words = Counter()
    for i in data:
        words.update(i)
    vocabulary = words.most_common(VOCABULARY_SIZE)
    words_list = [x[0] for x in vocabulary]
    words_to_indices = {x[0]: i + 1 for i, x in enumerate(vocabulary)}
    indices_to_words = {i + 1: x[0] for i, x in enumerate(vocabulary)}
    words_to_indices[u'*'] = VOCABULARY_SIZE + 1
    indices_to_words[VOCABULARY_SIZE + 1] = u'*'
    return words_list, words_to_indices, indices_to_words


def word_to_index(word, words_to_indices, word_list):
    if word in word_list:
        return words_to_indices[word]
    else:
        return len(word_list) + 1


def make_train_data(poem):
    pass


def get_style(poem):
    if len(poem) == 0:
        return 0
    style = len(poem[0])
    for sen in poem:
        if style != len(sen):
            return 0
    return style

if __name__ == "__main__":
    spb_5_X = []
    spb_5_Y = []
    cpb_5_X = []
    cpb_5_Y = []
    spb_7_X = []
    spb_7_Y = []
    cpb_7_X = []
    cpb_7_Y = []

    data = read_raw_data()
    words_list, words_to_indices, indices_to_words = make_word_indices(data)
    cPickle.dump([words_list, words_to_indices, indices_to_words], open(WORDS_INDICES_DICT_PATH, 'w+'))

    reg = '|'.join(puncts).replace('?', '\\?').replace('.', '\\.').replace('(', '\\(').replace(')', '\\)')

    cnt = 0
    for poem in data:
        if cnt % 1000 == 0:
            print 'poem ' + str(cnt)
        cnt += 1
        sens = re.split(reg, poem)
        indexed_sens = [[word_to_index(word, words_to_indices, words_list) for word in sen] for sen in sens]
        indexed_sens.pop(-1)
        style = get_style(indexed_sens)
        if style == 0:
            continue
        for line in range(len(indexed_sens)):
            if line % 2 == 1:
                X = indexed_sens[line - 1]
                Y = indexed_sens[line]
                if style == 5:
                    spb_5_X.append(X)
                    spb_5_Y.append(Y)
                elif style == 7:
                    spb_7_X.append(X)
                    spb_7_Y.append(Y)
            if line > 1:
                X = indexed_sens[line - 2] + indexed_sens[line - 1]
                Y = indexed_sens[line]
                if style == 5:
                    cpb_5_X.append(X)
                    cpb_5_Y.append(Y)
                elif style == 7:
                    cpb_7_X.append(X)
                    cpb_7_Y.append(Y)
    cPickle.dump([spb_5_X, spb_5_Y], open(POETRY_TRAIN_DATA_DIR+'spb_5', 'w+'))
    cPickle.dump([spb_7_X, spb_7_Y], open(POETRY_TRAIN_DATA_DIR+'spb_7', 'w+'))
    cPickle.dump([cpb_5_X, cpb_5_Y], open(POETRY_TRAIN_DATA_DIR+'cpb_5', 'w+'))
    cPickle.dump([cpb_7_X, cpb_7_Y], open(POETRY_TRAIN_DATA_DIR+'cpb_7', 'w+'))
