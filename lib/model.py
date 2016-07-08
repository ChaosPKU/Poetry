import os.path

from keras.models import Sequential
from keras.layers import Activation, Embedding
from seq2seq.models import AttentionSeq2seq, SimpleSeq2seq, Seq2seq
from config import *


def get_model(input_shape, output_length):
    model = Sequential()
    # model.add(Embedding(input_dim=VOCABULARY_SIZE + 1, output_dim=WORD_VECTOR_DIMENSION,
    #                     mask_zero=True, input_length=input_shape[1]))
    seq2seq = AttentionSeq2seq(
        batch_input_shape=input_shape,
        # input_dim=WORD_VECTOR_DIMENSION,
        # input_length=input_length,
        hidden_dim=HIDDEN_LAYER_DIMENSION,
        output_dim=VOCABULARY_SIZE + 2,
        output_length=output_length,
        bidirectional=True,
        depth=1
    )
    model.add(seq2seq)
    model.add(Activation('softmax'))
    return model

