# Poetry

## 1. Introduction

A RNN model to automatically generate Chinese ancient poems with the input of start words.
The idea is inspired by Weiyi Zheng's [tangshi-rnn](https://github.com/zhengwy888/tangshi-rnn) and Andrej Karpathy's [Char-RNN](https://github.com/karpathy/char-rnn).

## 2. Samples

```
1)             2)             3)             4)
白鹭窥鱼立，    一夜北风紧，    迟日江山丽，     去年今日此门中，
梅花晚少寒。    空山风雨寒。    无人入梦新。     白发春风未可怜。
青山犹出处，    山中有秋竹，    风霜独有路，     一笑一声天下去，
新月未无花。    风月更清香。    一月几人多。     自思春意又人心。
世事如收去，
相逢未自知。
```

## 3. Train and make poems

```
python main.py '一夜北风紧，' -p 100
```
Train a model If no available one exists or just load it and use the model to make poems.
Generate a word2vec model and use the result to init the embedding layer weights of the rnn model.
The model stacked 2 LSTM modules, each with 512 neurons. And 0.2 dropout rate while training. 
Please view the code to get more details.

## 4. Data

Only use a small part of Quan-Tang-Shi and Quan-Song-Shi.

## 5. Todo

A lot, such as adding rhythm.
