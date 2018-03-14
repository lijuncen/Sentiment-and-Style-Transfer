# -*- coding: gb18030 -*-
import string
import os
import math
import re

def load_emotion_word_dict(dirname):
    voca_dict = {} 
    voca_doc = {}
    num = 0
    fopen = open(dirname, 'r')
    for line in fopen:
        words = line.strip().decode('utf-8').split('\t')
        # print words
        w = words[0]
        # if w not in voca_dict:
        if len(words) == 2:
            voca_dict[w] = num
            voca_doc[w] = string.atoi(words[1])
            num += 1
    fopen.close()
    return voca_dict, voca_doc

def corpus2docmat(sen_array, voca_dict, ngram, voca_doc, total_num):
    mat_tf_idf = []
    sen_len = []
    for line in sen_array:
        # print len(sen_array)
        # print sen_array
        doc_vect = [0] * (len(voca_dict))
        lines = line.strip()
        # print lines
        # line=lines.decode('utf-8')
        for w in line:
            if(w in voca_dict.keys()):
                # print voca_dict[w]
                doc_vect[voca_dict[w]] += 1.0 * math.log(total_num / voca_doc[w]) / len(line)
                # doc_vect[voca_dict[w]] += 1.0/len(line)
        for w in range(0, len(line) - ngram + 1):
            if(line[w] + line[w + 1] in voca_dict.keys()):
                doc_vect[voca_dict[line[w] + line[w + 1]]] += 1.0 * math.log(total_num / voca_doc[line[w] + line[w + 1]]) / (len(line) - 1)
                # doc_vect[voca_dict[line[w]+line[w+1]]] += 1.0/(len(line)-1)
        mat_tf_idf.append(doc_vect)
        # doc_vect[-1]+=len(line)
        sen_len.append(len(line))
        # k="\t".join([str(j) for j in doc_vect])
        # f.write(k)
        # f.write("\n")
    return mat_tf_idf, sen_len
def load_no_sen_dict(dirname, thre_num):
    no_sen = []
    f = open(dirname, 'r')
    for line in f:
        lines = line.strip().split('\t')
        if(string.atoi(lines[1]) > thre_num):
            no_sen.append(lines[0].decode('utf-8'))
    f.close()
    return no_sen
def load_sense_dict(dirname):
    voca_dict = {}
    num = 0
    fopen = open(dirname, 'r')
    for line in fopen:
        w = line.strip().decode('utf-8')
        if w not in voca_dict:
            voca_dict[w] = num
            num += 1
    fopen.close()
    return voca_dict
def build_sense_vector(sen_array, voca_dict, ngram):
    mat_tf_idf = []
    for line in sen_array:
        doc_vect = [0] * (len(voca_dict))
        lines = line.strip()
        zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        m = zhPattern.findall(line)
        line = ''
        if(m):
            # line=m.group(0)
            # print m[0]
            for i in m:
                line += i
            # print line.encode('gb18030')
        for w in line:
            if(w in voca_dict.keys()):
                doc_vect[voca_dict[w]] += 1.0 / len(line)
        for w in range(0, len(line) - ngram + 1):
            if(line[w] + line[w + 1] in voca_dict.keys()):
                doc_vect[voca_dict[line[w] + line[w + 1]]] += 1.0 / (len(line) - 1)
        mat_tf_idf.append(doc_vect)
    return mat_tf_idf

ngram = 2
import pickle
ngram = int(ngram)
fn = os.path.split(os.path.realpath(__file__))[0] + '/judge_sense.pkl'
print fn
with open(fn, 'r') as f:
    sense_model = pickle.load(f)
f.close()
sense_dict = load_sense_dict(os.path.split(os.path.realpath(__file__))[0] + '/sense_dict.txt')

fn = os.path.split(os.path.realpath(__file__))[0] + '/nonsense_dict.txt'
weak_nonsense_set = set() 
strong_nonsense_set = set() 
with open(fn, 'r') as f:
    for line in f:
        line = line.decode('gb18030').strip().split('\t') 
        if line[1] == u'-1':
            strong_nonsense_set.add(line[0].strip())
        else:
            weak_nonsense_set.add(line[0].strip())
        
def isMakeSense(sentence):
    if sentence in strong_nonsense_set:
        return -1
    
    if sentence in weak_nonsense_set:
        return 0
    
    US_FLAG = 2
    if len(sentence) > 0:
        sen_array = []
        sen_array.append(sentence)
        US_FLAG = 2
        if(len(sentence) <= 40):
            if(len(sentence) < 40):
                sense_x = build_sense_vector(sen_array, sense_dict, ngram)
                sense_y = sense_model.predict(sense_x)
                US_FLAG = int(sense_y[0])
                return US_FLAG
#                         fw.write(sentence+str(US_FLAG)+'\n')

if __name__ == '__main__':
    print isMakeSense(u'不可以')
