#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: main.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>

from lib import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate new Chinese poems')
    parser.add_argument('start_words', type=str, help="Chinese characters in the beginning of the poem")
    parser.add_argument('-m', '--model', type=str, default=model_architecture_file_path,
                        help='Model architecture file path')
    parser.add_argument('-w', '--weight', type=str, default=model_weights_path, help='Model weights file path')
    parser.add_argument('-s', '--samples', type=int, default=nb_train_samples, help='Number of train samples')
    parser.add_argument('-e', '--epoch', type=int, default=nb_train_epoch, help='Number of train epoch')
    parser.add_argument('-f', '--files', type=int, default=nb_train_files,
                        help='Number of files in each training process')
    parser.add_argument('-r', '--row', type=int, default=nb_poem_rows, help='Number of poem rows')
    parser.add_argument('-c', '--col', type=int, default=nb_poem_cols, help='Number of poem cols')
    parser.add_argument('-b', '--batch', type=int, default=nb_batch_size, help='Batch size on training')
    args = parser.parse_args()

    model = get_model(args)

    outf = open(console_log_path, 'a')
    for i in xrange(1000):
        poem = make_poem(model, args.start_words[1:-1], words_to_indices, indices_to_words)
        outf.write('Poem %d \n' % i)
        outf.write(''.join(poem) + '\n')
    outf.close()
