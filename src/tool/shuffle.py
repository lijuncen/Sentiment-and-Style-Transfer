import sys
import random
f=open(sys.argv[1],'r')
fw=open(sys.argv[1]+'.shuffle','w')
sen_dict=[]
for line in f:
	sen_dict.append(line)
random.shuffle(sen_dict)
for i in sen_dict:
	fw.write(i)
f.close()
fw.close()

