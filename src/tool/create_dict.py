import sys
word_dict={}
f=open(sys.argv[1],'r')
fw=open(sys.argv[2],'w')
for line in f:
	lines=line.strip().split('\t')
	num=0
	for i in lines:
		if(num>=10):
			continue
		num+=1
		j=i.split(' ')
		for word in j:
			if(word !=''):
				if(word_dict.get(word)==None):
					word_dict[word]=1
				else:
					word_dict[word]+=1
dict1= sorted(word_dict.iteritems(), key=lambda d:d[1], reverse = True)
num=0
for i in dict1:
    if(i[1]>0 and num<30000):
        fw.write(i[0]+'\t'+str(num)+'\n')
        num+=1
