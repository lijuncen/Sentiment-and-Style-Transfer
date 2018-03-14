import sys
import string
f=open(sys.argv[1],'r')
fw=open(sys.argv[1]+'.final_result','w')
tag=sys.argv[2]
tmp_result=100
tmp=''
min_sen=''
for line in f:
	lines=line.strip().replace('<UNK> ',' ').replace(' <UNK>',' ').split('\t')
	if(lines[2]=='self'):
		continue
	if(tmp!=lines[1]):
		if(tmp!=''):
			fw.write(tmp+'\t'+min_sen+'\t'+tag+'\n')	
		tmp=lines[1]
		min_sen=lines[-3]
		tmp_result=string.atof(lines[-1])
	else:
		if(string.atof(lines[-1])<tmp_result):
			min_sen=lines[-3]
			tmp_result=string.atof(lines[-1])
fw.write(tmp+'\t'+min_sen+'\t'+tag+'\n')
		
