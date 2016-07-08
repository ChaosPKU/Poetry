# -*- encoding: utf-8 -*-

import sys
import cPickle
from config import *
from gensim.models import Word2Vec
from keras.models import model_from_json
from train import format_train_data
from lib.model import get_model
from lib.word2vector import get_word_vector
import numpy as np


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def sen2vec(sen, w2v_model):
    vec = np.zeros((1, len(sen), WORD_VECTOR_DIMENSION), dtype=np.float)
    for i, word in enumerate(sen):
        vec[0, i] = get_word_vector(word, w2v_model)
    return vec


def gen_sen_1():
    sen = '塞外悲风切'.decode('utf-8')
    return sen


def gen_next_sen(model, sen_vec, indices_to_words):
    predictions = model.predict(sen_vec)[0]
    next_sen = u''
    for vec in predictions:
        # max = 0
        # for index, val in enumerate(vec):
        #     if val > max:
        #         max = val
        #         i = index
        # next_sen += indices_to_words[i]
        next_sen += indices_to_words[sample(vec)]
    return next_sen


def gen_poem(indices_to_words, w2v_model, spb, cpb):
    print 'generating ...'
    sen_1 = gen_sen_1()
    sen_2 = gen_next_sen(spb, sen2vec(sen_1, w2v_model), indices_to_words)
    sen_3 = gen_next_sen(cpb, sen2vec(sen_1 + sen_2, w2v_model), indices_to_words)
    sen_4 = gen_next_sen(cpb, sen2vec(sen_2 + sen_3, w2v_model), indices_to_words)

    print sen_1, sen_2, sen_3, sen_4

if __name__ == '__main__':
    print 'loading ...'
    words_list, words_to_indices, indices_to_words = cPickle.load(open(WORDS_INDICES_DICT_PATH, 'r'))
    print '1'
    w2v_model = Word2Vec.load(W2V_MODEL_PATH)
    print '2'
    spb = cPickle.load(open(MODEL_ARCHITECTRUE_PATH + 'spb_5.model', 'r'))
    print '3'
    cpb = cPickle.load(open(MODEL_ARCHITECTRUE_PATH + 'cpb_5.model', 'r'))
    print '4'
    for i in range(100):
        gen_poem(indices_to_words, w2v_model, spb, cpb)

