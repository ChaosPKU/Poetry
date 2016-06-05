#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: config.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>

input_max_len = 1
poem_division_step = 1
nb_train_text_words = 1000

puncts = u"。.？?！!，,、；;：:（）《》<>()"
poetry_raw_directory_path = "../data/raw_data/raw_poem_all/"
poetry_raw_data_path = '../data/raw_data/short_poetry.txt'
poetry_vocabulary_data_path = '../data/gen_data/poetry.txt'
poetry_gen_data_model_path = '../data/gen_data/poetry.model'
poetry_gen_data_vector_path = '../data/gen_data/poetry.vector'

word_vector_dimension = 100
word_min_count = 10
word_predicted_window = 10

model_weights_path = "../data/model_data/model_weights"
model_batch_size = 1000
model_input_nb_words = 1
model_output_nb_words = 20
model_input_dim = model_input_nb_words * word_vector_dimension
model_output_dim = model_output_nb_words * word_vector_dimension
model_hidden_dim = 200
model_activation = 'relu'
model_inner_activation = 'relu'
model_output_activation = 'relu'
model_nb_samples = 10000
model_dropout = 0.5
model_loss = 'sparse_categorical_crossentropy'
model_optimizer = 'RMSprop'
model_nb_epoch = 60

