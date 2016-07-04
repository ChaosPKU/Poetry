#!/usr/bin/env bash

# configure the running environment
pip install -r requirements.txt
cd bin
# generate word2vec model
python word2vec.py
# generate train data
python preprocessing.py
# train a new model or use existed model to make poems
python main.py '' -p 100