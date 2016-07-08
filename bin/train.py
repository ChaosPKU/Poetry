import sys
from lib.model import get_model
from lib.word2vector import get_word_vector
from config import *
from gensim.models import Word2Vec
import cPickle
import numpy as np


def format_train_data(file_name, input_length, output_length):
    src_seq, dst_seq = cPickle.load(open(POETRY_TRAIN_DATA_DIR + file_name, 'r'))
    X = np.zeros((len(src_seq), input_length, WORD_VECTOR_DIMENSION), dtype=np.float)
    Y = np.zeros((len(dst_seq), output_length, VOCABULARY_SIZE + 2), dtype=np.bool)
    for i, seq in enumerate(src_seq):
        for j, word in enumerate(seq):
            X[i, j] = get_word_vector(indices_to_words[word], w2v_model)
    for i, seq in enumerate(dst_seq):
        for j, word in enumerate(seq):
            Y[i, j, word] = 1
    return X, Y


def train(model, X, Y, model_name):
    print 'compile model ...'
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    print 'training ...'
    model.fit(X, Y, batch_size=128, nb_epoch=100, verbose=1)
    model.predict(X[0:1])
    print 'save model ...'
    # outf = open(MODEL_ARCHITECTRUE_PATH + model_name + '.json', 'w')
    # outf.write(model.to_json())
    # outf.close()
    cPickle.dump(model, open(MODEL_ARCHITECTRUE_PATH + model_name + '.model', 'w'))
    # model.save_weights(MODEL_WEIGHTS_PATH + model_name + '.h5', overwrite=True)
    print 'done.'

if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    words_list, words_to_indices, indices_to_words = cPickle.load(open(WORDS_INDICES_DICT_PATH, 'r'))
    w2v_model = Word2Vec.load(W2V_MODEL_PATH)

    # spb_5
    X, Y = format_train_data('spb_5', 5, 5)
    train(get_model((1, 5, WORD_VECTOR_DIMENSION), 5), X, Y, 'spb_5')

    # cpb_5
    X, Y = format_train_data('cpb_5', 10, 5)
    train(get_model((1, 10, WORD_VECTOR_DIMENSION), 5), X, Y, 'cpb_5')

    # spb_7
    X, Y = format_train_data('spb_7', 7, 7)
    train(get_model((1, 7, WORD_VECTOR_DIMENSION), 7), X, Y, 'spb_7')

    # cpb_7
    X, Y = format_train_data('cpb_7', 14, 7)
    train(get_model((1, 14, WORD_VECTOR_DIMENSION), 7), X, Y, 'cpb_7')
