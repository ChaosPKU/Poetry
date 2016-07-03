## First Sentence Generator

### 1. Introduction

A python file of generating the first sentence with the input of user:

* 5-char quatrain or 7-char quatrain
* the subject of this sentence based on the category of *ShiXueHanYing*

### 2. Important API

* `find_best_sentences(n=10)`
 
	The hint of user input is given in this function, and based on user input, generate n-best sentences with format of [poetry, score].

* `read_character_tone()`

	Get a character-tone dictionary based on *PingShuiYun*.

* `judge_tonal_pattern(rows, chars)`
	
	Get the tonal pattern number for a 5-char sentence or 7-char pattern, if this sentence does not follow any of the 4 patterns, return -1.

### 3. Dependency

[SRILM - The SRI Language Modeling Toolkit](http://www.speech.sri.com/projects/srilm/)





