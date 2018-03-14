import sys
sen_dict1={}
f=open(sys.argv[1],'r')
for line in f:
	lines=line.strip().split('\t')
	if(sen_dict1.get(lines[1])==None):
		sen_dict1[lines[1]]=lines[0]
f.close()
f=open(sys.argv[2],'r')
fw=open(sys.argv[2]+'.change','w')
tmp=''
for line in f:
	#fw.write(line)
	lines=line.split('\t')
	
	if(lines[1]!=tmp and lines[0]!=sen_dict1[lines[0]]):
		lines1=line.split('\t')
		lines1[0]=sen_dict1[lines1[0]]
		lines1[2]='self'
		#fw.write('\t'.join(lines1[:]))
	
	if(sen_dict1.get(lines[0])!=None):
		lines[0]=sen_dict1[lines[0]]
	else:
		print lines[0]
	tmp=lines[1]
	fw.write('\t'.join(lines[:]))
f.close()
fw.close()
