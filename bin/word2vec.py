#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: word2vec.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>

import sys
# import jieba
from configs.config import *
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing


# def build_vocabulary():
#     outp = open(poetry_vocabulary_data_path, 'w')
#     with open(poetry_raw_data_path, 'r') as inp:
#         lines = inp.readlines()
#         for line in lines:
#             newline = ' '.join(' '.join(line.split(u'。')).split(u'，'))
#             outp.write(' '.join(jieba.cut(newline, cut_all=True)))
#     outp.close()


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf-8')
    # build_vocabulary()
    model = Word2Vec(
        LineSentence(poetry_vocabulary_data_path),
        size=word_vector_dimension,
        min_count=word_min_count,
        window=word_predicted_window,
        workers=multiprocessing.cpu_count())
    model.save(poetry_gen_data_model_path)
    model.save_word2vec_format(poetry_gen_data_vector_path, binary=False)

