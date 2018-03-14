import sys
import time
from scipy import spatial
import numpy as np
import string
main_data=sys.argv[2]
main_function=sys.argv[3]
def cosine_simi(a,b):
	result = 1 - spatial.distance.cosine(a,b)
	return result
def process_line(line):
	lines=line.strip().split('\t')
	sen='\t'.join(lines[:-1])
	nums=lines[-1].split(' ')
	tmp_array=[]
	for i in nums:
		tmp_array.append(string.atof(i))
	return sen,tmp_array

sens=[]
arrs=[]
f1=open('sentiment.train.'+str(1-string.atoi(sys.argv[1]))+'.template.'+main_function+'.emb','r')
for line in f1:
	sen2,arr2=process_line(line)
	sens.append(sen2)
	arrs.append(arr2)
f1.close()

f=open('sentiment.test.'+sys.argv[1]+'.template.'+main_function+'.emb','r')
fw=open('sentiment.test.'+sys.argv[1]+'.template.'+main_function+'.emb.result','w')
for line in f:
	sen1,arr1=process_line(line)
	tmp_sen_score_dict={}
	if(main_function=='label'):
		sen1s=sen1.split('\t')
		sen1s[-2]=str(1-string.atoi(sys.argv[1]))
		sen1c='\t'.join(sen1s)
		fw.write(sen1+'\t'+'1'+'\t'+sen1c+'\n')
		continue
	for i in range(len(sens)):
		#sen2,arr2=process_line(line2)
		result1=cosine_simi(arr1,arrs[i])
		tmp_sen_score_dict[sens[i]]=result1
	dict1=sorted(tmp_sen_score_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
	for j in range(min(10,len(dict1))):
		fw.write(sen1+'\t'+str(dict1[j][1])+'\t'+str(dict1[j][0])+'\n')
f.close()
fw.close()

