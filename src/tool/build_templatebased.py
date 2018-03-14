import sys
f=open(sys.argv[1],'r')
fw=open(sys.argv[1]+'.result','w')
tmp=''
def write_sen(sens,tmp1):
	tmp=''
	num=0
	for i in sens:
		if(i!='slotholdplace'):	
			tmp+=i+' '
		else:
			if(num<len(tmp1)):
				tmp+=tmp1[num]+' '
				num+=1
			else:
				pass
		#tmp+=' '
	tmp=tmp.strip()
	tmp2=tmp[:-2]
	while num<len(tmp1):
		tmp2+=' '+tmp1[num]+' '
		num+=1
	return tmp2.strip()+tmp[-2:]
rule_dict={}
for line in f:
	lines=line.strip().split('\t')
	if('.' in lines[3]):
		continue
	if(tmp!=lines[0]):
		tmp=lines[0]
		tmp1=lines[1].split('######')
		tmp2=lines[3].split('######')
		words=lines[0].split(' ')
		if(tmp1[0]=='SELF'):
			sen1=tmp
		else:
			sen1=write_sen(words,tmp1)
		sen2=write_sen(words,tmp2)
		rule_dict[sen1]=sen2
f.close()
f=open(sys.argv[2],'r')
for line in f:
	line=line.strip()
	if(rule_dict.get(line)!=None):
		fw.write(line+'\t'+rule_dict.get(line)+'\t'+sys.argv[2][-1]+'\n')
	else:
		fw.write(line+'\t'+line+'\t'+sys.argv[2][-1]+'\n')
fw.close()
