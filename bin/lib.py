#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: lib.py
# Date: 2016-06-04
# Author: Chaos <xinchaoxt@gmail.com>

from configs.config import *
# from gensim.models import Word2Vec
import numpy as np

# import re
# from model.model import LSTM_RNN_Model
# import random
#
# word2vec_model = Word2Vec.load(poetry_gen_data_model_path)
# vocab_size = len(word2vec_model.vocab)
#
#
# def get_word_vector(word):
#     if word in word2vec_model.vocab.keys():
#         return word2vec_model[word]
#     else:
#         return [0] * word_vector_dimension
#
#
# def load_data(nb_samples, X_word, Y_word):
#     X = []
#     Y = []
#     inf = open(poetry_vocabulary_data_path, 'r')
#     cnt = 0
#     for line in inf:
#         cnt += 1
#         sens = re.split(' |\t', line.decode('utf-8').strip())
#         X.append(np.array([get_word_vector(x) for x in sens[:X_word]]).flatten())
#         Y.append(np.array([get_word_vector(x) for x in sens[:Y_word]]).flatten())
#         if cnt == nb_samples:
#             break
#     return np.array(X), np.array(Y)
#
#
# def get_vocabulary_list():
#     return word2vec_model.vocab.keys()
#
#
# def get_data():
#     inf = open(poetry_vocabulary_data_path, 'r')
#     train_text = []
#     for line in inf:
#         for sentence in line.split('\t'):
#             train_text += sentence.split()
#         if len(train_text) > model_nb_samples:
#             break
#     train_text = train_text[:model_nb_samples + 1]
#     words = set(train_text)
#     word_to_indices = dict((word, idx) for idx, word in enumerate(words))
#     indices_to_word = dict((idx, word) for idx, word in enumerate(words))
#     X = np.zeros((len(train_text), input_max_len, len(words)), dtype=np.float32)
#     Y = np.zeros((len(train_text), len(words)), dtype=np.float32)
#     for i, word in enumerate(train_text[:-1]):
#         X[i, 0, word_to_indices[word]] = 1
#         Y[i, word_to_indices[train_text[i + 1]]] = 1
#     return X, Y, word_to_indices, indices_to_word, len(words), train_text
#
#
# def lstm_train():
#     X, Y, word_to_indices, indices_to_word, nb_words, train_text = get_data()
#     lstm = LSTM_RNN_Model(nb_words, 100, nb_words)
#     lstm.build(dropout=model_dropout)
#     outf = open(console_log_path, 'a')
#     outf.write("Vocabulary Length: %d, Text Length: %d" % (nb_words, len(train_text)) + '\n')
#     for iter in xrange(model_nb_epoch):
#         outf.write("==============================================================" + '\n')
#         outf.write("Iteration: " + str(iter) + '\n')
#         lstm.model.fit(X, Y, batch_size=model_batch_size, nb_epoch=1)
#         start_index = random.randint(0, len(train_text) - 1)
#         result = [train_text[start_index]]
#         for i in xrange(27):
#             seed = np.zeros((1, 1, nb_words))
#             seed[0, 0, word_to_indices[result[-1]]] = 1
#             predictions = lstm.model.predict(seed, verbose=0)[0]
#             next_index = sample(predictions)
#             next_word = indices_to_word[next_index]
#             result.append(next_word)
#         # print ''.join(result)
#         outf.write(''.join(result) + '\n')
#     outf.close()
#
#


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

    for i in xrange(poem_max_length):
        inputs = np.asarray([text[-poem_max_length:]])
        preds = model.predict(inputs, verbose=0)[0]
        next_index = sample(preds, sample_temperature) + 1
        text.append(next_index)
        if next_index == words_to_indices['$'.decode('utf-8')]:
            break

    return [indices_to_words[i].encode("utf-8") for i in text[poem_max_length + 1:-1]]

