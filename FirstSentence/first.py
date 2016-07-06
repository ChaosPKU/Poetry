## coding=utf-8 ##
import os
import operator
import random


# 0 represents either, 1 represents Ping, -1 represents Ze
FIVE_PINGZE = [[[0, -1, 1, 1, -1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],
               [[0, -1, -1, 1, 1], [1, 1, -1, -1, 1], [0, 1, 1, -1, -1], [0, -1, -1, 1, 1]],
               [[0, 1, 1, -1, -1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1]],
               [[1, 1, -1, -1, 1], [0, -1, -1, 1, 1], [0, -1, 1, 1, -1], [1, 1, -1, -1, 1]]]

SEVEN_PINGZE = [[[0, 1, 0, -1, -1, 1, 1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],
                [[0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1], [0, -1, 0, 1, 1, -1, -1], [0, 1, 0, -1, -1, 1, 1]],
                [[0, -1, 1, 1, -1, -1, 1], [0, 1, 0, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]],
                [[0, -1, 0, 1, 1, -1, -1], [0, 0, -1, -1, -1, 1, 1], [0, 1, 0, -1, 1, 1, -1], [0, -1, 1, 1, -1, -1, 1]]]

# Pingshui Rhyme contains the Ping/Ze tonal of all characters
PINGSHUI_PATH = "./dataset/pingshui.txt"
# Shixuehanying contains the category of words
SHIXUEHANYING_PATH = "./dataset/shixuehanying.txt"
# n-gram executing file path (need to install srilm first)
NGRAM_PATH = "/home/user/poem/srilm/bin/i686-m64/ngram"
# candidates file name
CANDIDATE_FILE_PATH = './intermediate/candidates/candidates.'
# candidates score file name
CANDIDATE_SCORE_PATH = "./intermediate/score/score.ppl."
# top candidates file name
TOP_RESULT_PATH = "./intermediate/top/top."
# language model file name
LANGUAGE_MODEL_PATH = "./poem_lm/first.poem.lm"


# get the tonal dictionary of each character based on "Pingshui Rhyme"
def read_character_tone():
    tonals = {}
    f = open(PINGSHUI_PATH, 'r')
    isping = False
    while True:
        line = f.readline()
        line = line.strip()
        if line:
            if line[0] == '/':
                isping = not isping
                continue
            line = unicode(line, "utf-8")
            for i in line:
                tonals[i] = isping
        else:
            break
    return tonals


# get the words list of shixuehanying based on "Shi Xue Han Ying"
def read_shixuehanying():
    f = open(SHIXUEHANYING_PATH, 'r')
    categories = []
    labels = []
    words = []
    while True:
        line = f.readline()
        line = line.strip()
        if line:
            line = unicode(line, "utf-8")
            if line[0] == '<':
                if line[1] == 'b':
                    titles = line.split('\t')
                    categories.append(titles[2])
            else:
                line = line.split('\t')
                if len(line) == 3:
                    tmp = line[2].split(' ')
                    tmp.append(line[1])
                else:
                    tmp = line[1]
                if len(tmp) >= 10:
                    labels.append(categories[len(categories) - 1] + "-" + line[1])
                    words.append(tmp)
        else:
            break
    return categories, labels, words


# main function of getting user input of keywords, tonal pattern and 5/7 mark
def user_input():
    categories, labels, words = read_shixuehanying()
    chars = label = 0
    while True:
        print "Please choose poem structure:\n5. 5-char quatrain\n7. 7-char quatrain\n"
        chars = input()
        if chars == 5 or chars == 7:
            break
        else:
            print "Wrong input, Please try again"
    while True:
        print "Please choose poem subject:\n"
        print u'0 随机'
        for i in range(0, len(labels)):
            print str(i + 1) + " " + labels[i]
        label = input()
        if label == 0:
            label = int(random.uniform(1, len(labels)))
            break
        if not 1 <= label <= len(labels):
            print "Wrong input, Please try again"
            continue
        else:
            break
    print "User choose " + str(chars) + "-char quatrain with subject " + \
          labels[label - 1]
    return words[label - 1], chars, labels[label - 1]


# judge if the given sentence follows the given tonal pattern
def judge_tonal_pattern(row, chars):
    tonal_hash = read_character_tone()
    # remove poem with duplicated characters
    if len(row) != len(set(row)):
        return -1
    # judge rhythm availability
    if chars == 5:
        tone = FIVE_PINGZE
    else:
        tone = SEVEN_PINGZE
    for i in range(0, 4):
        for j in range(0, chars + 1):
            if j == chars:
                return i
            if tone[i][0][j] == 0:
                continue
            elif tone[i][0][j] == 1 and (not row[i] in tonal_hash or tonal_hash[row[i]] == True):
                continue
            elif tone[i][0][j] == -1 and (not row[i] in tonal_hash or tonal_hash[row[i]] == False):
                continue
            else:
                break
    return -1


# based on user option of chars, tonal and keywords, generate all possible candidates of the first sentence
def generate_all_candidates():
    vec, chars, subject = user_input()
    candidates = []
    result = []
    vec.sort(key=lambda x: len(x))
    indexes = [0, 0, 0, 0, 0, 0]
    for i in range(0, len(vec)):
        indexes[len(vec[i])] += 1
    for i in range(1, len(indexes)):
        indexes[i] += indexes[i - 1]
    if chars == 5:
        # 5
        for i in range(indexes[4], len(vec)):
            candidates.append(vec[i])
        # 4-1
        for i in range(0, indexes[1]):
            for j in range(indexes[3], indexes[4]):
                candidates.append(vec[i] + vec[j])
                candidates.append(vec[j] + vec[i])
        # 3-2
        for i in range(indexes[1], indexes[2]):
            for j in range(indexes[2], indexes[3]):
                candidates.append(vec[i] + vec[j])
                candidates.append(vec[j] + vec[i])
        # 2-2-1
        for i in range(indexes[1], indexes[2]):
            for j in range(i + 1, indexes[2]):
                for k in range(0, indexes[1]):
                    candidates.append(vec[i] + vec[j] + vec[k])
                    candidates.append(vec[i] + vec[k] + vec[j])
                    candidates.append(vec[j] + vec[i] + vec[k])
                    candidates.append(vec[j] + vec[k] + vec[i])
                    candidates.append(vec[k] + vec[j] + vec[i])
                    candidates.append(vec[k] + vec[i] + vec[j])
    elif chars == 7:
        # 5-2
        for i in range(indexes[4], len(vec)):
            for j in range(indexes[1], indexes[2]):
                candidates.append(vec[i] + vec[j])
                candidates.append(vec[j] + vec[i])
        # 4-3
        for i in range(indexes[3], indexes[4]):
            for j in range(indexes[2], indexes[3]):
                candidates.append(vec[i] + vec[j])
                candidates.append(vec[j] + vec[i])
        # 3-3-1
        for i in range(indexes[2], indexes[3]):
            for j in range(i + 1, indexes[3]):
                for k in range(0, indexes[1]):
                    candidates.append(vec[i] + vec[j] + vec[k])
                    candidates.append(vec[i] + vec[k] + vec[j])
                    candidates.append(vec[j] + vec[i] + vec[k])
                    candidates.append(vec[j] + vec[k] + vec[i])
                    candidates.append(vec[k] + vec[j] + vec[i])
                    candidates.append(vec[k] + vec[i] + vec[j])
        # 3-2-2
        for i in range(indexes[1], indexes[2]):
            for j in range(i + 1, indexes[2]):
                for k in range(indexes[2], indexes[3]):
                    candidates.append(vec[i] + vec[j] + vec[k])
                    candidates.append(vec[i] + vec[k] + vec[j])
                    candidates.append(vec[j] + vec[i] + vec[k])
                    candidates.append(vec[j] + vec[k] + vec[i])
                    candidates.append(vec[k] + vec[j] + vec[i])
                    candidates.append(vec[k] + vec[i] + vec[j])
    # select candidates with given tonal patterns
    for i in candidates:
        j = judge_tonal_pattern(i, chars)
        if j == -1:
            continue
        result.append(i)
    return result, subject


# select one of the best candidates by randomly select one from top n results
def find_best_sentences(n=10):
    candidates = []
    subject = ""
    while True:
        candidates, subject = generate_all_candidates()
        if len(candidates) != 0:
            break
        else:
            print "There is not enough sentence in this category, please choose another one"

    # write the candidates into the candidates file
    subject = subject.encode("utf-8")
    output = open(CANDIDATE_FILE_PATH + subject, 'w')
    for string in candidates:
        for j in range(0, len(candidates[0])):
            tmp = string[j].encode("utf-8")
            output.write(tmp + " ")
        output.write("\n")
    output.close()

    # use n-gram model to find the best sentence
    cmd = '%s -ppl %s -debug 1 -order 3 -lm %s > %s' % \
          (NGRAM_PATH, CANDIDATE_FILE_PATH + subject, LANGUAGE_MODEL_PATH, CANDIDATE_SCORE_PATH + subject)
    os.system(cmd)

    # find n-best sentences with largest score
    fp = open(CANDIDATE_SCORE_PATH + subject)
    candidates_score_list = []
    for line in fp.readlines():
        if 'file' in line:
            break
        if 'words' in line or line == '\n':
            continue
        elif 'ppl' in line:
            tmp.append(float(line.strip('\r\n').split(' ')[-1]))
            candidates_score_list.append(tmp)
        else:
            tmp = [line.strip()]
    candidates_score_list.sort(key=operator.itemgetter(1))
    result = candidates_score_list[0: min(n, len(candidates_score_list))]

    # write the result to file
    output = open(TOP_RESULT_PATH + subject, 'w')
    print "Top " + str(n) + " results:"
    for i in result:
        s = i[0].replace(' ', '')
        output.write(s + "\n")
        print s
    output.close()
    res = random.choice(result)[0]
    res = res.replace(' ', '')
    print "The first sentence is: " + res
    return res

find_best_sentences(10)



