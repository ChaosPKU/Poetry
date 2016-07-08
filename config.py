# -*- encoding: utf-8 -*-

puncts = u'。.？?！!，,、；;：:（）《》<>()'

RAW_DATA_DIR = '../data/raw_data/raw_poem_all/'
RAW_DATA_FILES = ['qss_tab.txt', 'qts_tab.txt', 'qtais_tab.txt']
WORDS_INDICES_DICT_PATH = '../data/gen_data/indices.pkl'
EMBEDDING_WEIGHTS_PATH = '../data/gen_data/embedding_weights'
POETRY_TRAIN_DATA_DIR = '../data/gen_data/train_data/'
MODEL_ARCHITECTRUE_PATH = "../data/model_data/"
MODEL_WEIGHTS_PATH = "../data/model_data/"
W2V_DATA_PATH = '../data/gen_data/w2v.data'
W2V_MODEL_PATH = '../data/gen_data/w2v.model'

VOCABULARY_SIZE = 5000
WORD_VECTOR_DIMENSION = 500
WORD_MIN_COUNT = 5
WORD_PREDICTED_WINDOW = 10

# TOKEN_REPRESENTATION_SIZE =
# INPUT_SEQUENCE_LENGTH =
HIDDEN_LAYER_DIMENSION = 200
# VOCABULARY_SIZE =
# ANSWER_MAX_TOKEN_LENGTH =