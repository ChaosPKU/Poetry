#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: config.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>

poem_max_length = 32
poem_division_step = 1
sample_temperature = 0.5
nb_train_samples = 100000
nb_train_epoch = 100
nb_train_files = 10
vocabulary_data_size = 5000
nb_poem_rows = 4
nb_poem_cols = 7
nb_batch_size = 128
nb_make_poems = 1000


puncts = u"。.？?！!，,、；;：:（）《》<>()"
tags = u"^$"
poetry_raw_directory_path = "../data/raw_data/raw_poem_all/"
short_poetry_raw_data_path = '../data/raw_data/short_poetry.txt'
poetry_vocabulary_data_path = '../data/gen_data/poetry.txt'
poetry_gen_data_model_path = '../data/gen_data/poetry.model'
poetry_gen_data_vector_path = '../data/gen_data/poetry.vector'
poem_raw_file_name = ["qss_tab.txt", "qts_tab.txt", "qtais_tab.txt"]

word_vector_dimension = 500
word_min_count = 5
word_predicted_window = 10

words_indices_dict_path = '../data/gen_data/indices.pkl'
poetry_train_data_dir = '../data/gen_data/train_data/'
poetry_train_data_path = '../data/gen_data/train_data/train_data.pkl'
model_architecture_file_path = "../data/model_data/model.json"
model_weights_path = "../data/model_data/model_weights.h5"

log_path = '../log/'
console_log_path = log_path + 'console.log'