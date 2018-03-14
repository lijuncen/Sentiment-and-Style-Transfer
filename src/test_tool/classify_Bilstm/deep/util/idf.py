# -*- coding: utf-8 -*-
import codecs
import string
import os
import math
_idf_dict = dict()
_answer_df_dict = dict()

with codecs.open(os.path.split(os.path.realpath(__file__))[0] + \
                 '/idf', 'r', 'gb18030', 'ignore') as f:
    for line in f:
        line = line.strip().split(u'\t')
        if len(line) < 2:
            continue
        word, idf = line
        idf = string.atof(idf)
        _idf_dict[word] = idf

with codecs.open(os.path.split(os.path.realpath(__file__))[0] + \
                 '/answer_idf', 'r', 'gb18030', 'ignore') as f:
    for line in f:
        line = line.strip().split(u'\t')
        if len(line) < 2:
            continue
        word, idf = line
        idf = string.atof(idf)
        _answer_df_dict[word] = idf
        
def get_idf(word):
    if word in _idf_dict:
        return _idf_dict[word]
    else:
        return 1
    
def get_answer_df(answer):
    if answer in _answer_df_dict:
        return math.log(25349539/ _answer_df_dict[answer])
    else:
        return 1

if __name__ == '__main__':
    print get_idf(u'成都是屌丝功')
