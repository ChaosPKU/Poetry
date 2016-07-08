import sys
from config import *
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import numpy as np


def uwrite(fout, text, encode='utf-8'):
    fout.write(text.encode(encode))


def gen_data(outFile):
    outf = open(outFile, 'a')
    for file in RAW_DATA_FILES:
        inf = open(RAW_DATA_DIR + file)
        fields = inf.readline().decode("utf-8").strip().split('\t')
        body_idx = fields.index(u'body')
        for line in inf:
            fields = line.decode('utf-8').strip().split('\t')
            body = fields[body_idx]
            newsens = [' '.join(word) for word in body if len(word) != 0]
            newline = '\t'.join(newsens) + '\n'
            if len(newline.strip()) > 0:
                uwrite(outf, newline)
        inf.close()
    outf.close()


def train_w2v():
    model = Word2Vec(
        LineSentence(W2V_DATA_PATH),
        size=WORD_VECTOR_DIMENSION,
        min_count=WORD_MIN_COUNT,
        window=WORD_PREDICTED_WINDOW,
        workers=multiprocessing.cpu_count())
    model.save(W2V_MODEL_PATH)
    model.save_word2vec_format(W2V_DATA_PATH, binary=False)


def get_word_vector(word, model):
    if word in model.vocab:
        return np.array(model[word])

    # return a zero vector for the words that are not presented in the model
    return np.zeros(WORD_VECTOR_DIMENSION)


def get_vectorized_word_sequence(sequence, model, max_sequence_length, reverse=False):
    vectorized_word_sequence = np.zeros((max_sequence_length, WORD_VECTOR_DIMENSION), dtype=np.float)

    for i, word in enumerate(sequence):
        vectorized_word_sequence[i] = get_word_vector(word, model)

    if reverse:
        vectorized_token_sequence = vectorized_word_sequence[::-1]

    return vectorized_word_sequence

if __name__ == '__main__':
    pass
