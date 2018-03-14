# coding=utf-8
# -*- coding: utf-8 -*-
import os
__keywordList = None

def __init():
    import codecs
    globals()['__keywordList'] = list()
    with codecs.open(os.path.split(os.path.realpath(__file__))[0] + \
                     "/rubbish.dat", "r", "utf-8", "ignore") as f:
        for line in f:
            line = line.strip()
            globals()['__keywordList'].append(line)

def isRubbishSentence(sentence):
    """
        This method filter sentence and denote whether
        the given sentence is rubbish.
        The rubbish word list is in "util/rubbish.dat"
    """
    
    if globals()['__keywordList'] == None:
        __init()
    flag = False
    for s in globals()['__keywordList']:
        if s in sentence:
            flag = True
            break
    return flag

