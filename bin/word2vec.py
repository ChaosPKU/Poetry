#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: word2vec.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>

import sys
from configs.config import *
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing


def uwrite(fout, text, encode='utf-8'):
    fout.write(text.encode(encode))


def gen_data(outFile):
    outf = open(outFile, 'a')
    for file in poem_raw_file_name:
        inf = open(poetry_raw_directory_path + file)
        fields = inf.readline().decode("utf-8").strip().split('\t')
        body_idx = fields.index('body')
        for line in inf:
            fields = line.decode('utf-8').strip().split('\t')
            body = "^" + fields[body_idx] + "$"
            newsens = [' '.join(word) for word in body if len(word) != 0]
            newline = '\t'.join(newsens) + '\n'
            if len(newline.strip()) > 0:
                uwrite(outf, newline)
        inf.close()
    outf.close()

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    gen_data(poetry_vocabulary_data_path)
    model = Word2Vec(
        LineSentence(poetry_vocabulary_data_path),
        size=word_vector_dimension,
        min_count=word_min_count,
        window=word_predicted_window,
        workers=multiprocessing.cpu_count())
    model.save(poetry_gen_data_model_path)
    model.save_word2vec_format(poetry_gen_data_vector_path, binary=False)

