import sys
for i in range(2):
	f=open('sentiment.test.'+str(i)+'.template.orgin.emb.result.filter.result','r')
	fw=open('sentiment.test.'+str(i)+'.retrieval','w')
	tmp=''
	tmp_array=[]
	for line in f:
		lines=line.strip().split('\t')
		if(tmp!=lines[1] and tmp!=''):
			fw.write(tmp+'\t'+tmp_array[0]+'\t'+str(i)+'\n')
			tmp_array=[lines[4]]
		else:
			tmp_array.append(lines[4])
		tmp=lines[1]
	fw.write(tmp+'\t'+tmp_array[0]+'\t'+str(i)+'\n')	
