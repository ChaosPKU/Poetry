#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
# File: data2word.py
# Date: 2016-06-02
# Author: Chaos <xinchaoxt@gmail.com>

import os
import sys
import re
from configs.config import *


# def matches(reg, s):
#     m = re.match(reg, s)
#     if m is None or m.group() != s:
#         return None
#     return m


def uwrite(fout, text, encode='utf-8'):
    fout.write(text.encode(encode))


def gen_data(inFile, outFile):
    inf = open(inFile)
    outf = open(outFile, 'a')
    fields = inf.readline().decode("utf-8").strip().split('\t')
    body_idx = fields.index('body')
    reg = '|'.join(puncts).replace("?", "\\?").replace(
        ".", "\\.").replace("(", "\\(").replace(")", "\\)")
    for line in inf:
        fields = line.decode('utf-8').strip().split('\t')
        body = fields[body_idx]
        sens = re.split(reg, body)
        newsens = [' '.join(word) for word in sens if len(word) != 0]
        newline = '\t'.join(newsens) + '\n'
        if len(newline.strip()) > 0:
            uwrite(outf, newline)
    outf.close()


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    file_list = os.listdir(poetry_raw_directory_path)
    for file in file_list:
        gen_data(poetry_raw_directory_path + file, poetry_vocabulary_data_path)
