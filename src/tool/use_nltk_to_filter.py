import sys
import string
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import re
f=open(sys.argv[1],'r')
fw=open(sys.argv[1]+'.filter','w')
nltk.data.path.append("nltk_data/")
for line in f:
	lines=line.strip().split('\t')
	#if(len(lines)!=2 or 'play' in line):
	if(len(lines)!=2):
		continue
	if(string.atof(lines[1])<5.0):
		break
	ne_tree = ne_chunk(pos_tag(word_tokenize(lines[0])))
	iob_tagged = tree2conlltags(ne_tree)
	#print iob_tagged
	flag=0
	for ll in iob_tagged:
		if('NN' in ll[1]):
			flag=1
			break
	if(flag==0):
		fw.write(line)	
