import sys
f=open(sys.argv[1],'r')
fw=open(sys.argv[1]+'.filter','w')
num=0
tmp=''
tmp_sen=''
already_sen={}
for line in f:
	lines=line.strip().split('\t')
	if(num>15000):
		break
	#fw.write(lines[0]+'\t'+lines[1]+'\t'+lines[2]+'\t'+lines[3]+'\t'+lines[-3]+'\t'+lines[4]+'\n')
	if(tmp!=lines[0] and lines[0]!=lines[1] and 'label' not in sys.argv[1]):
		pass
		#fw.write(lines[0]+'\t'+lines[1]+'\t'+'self'+'\t'+lines[3]+'\t'+lines[-3]+'\t'+lines[4]+'\n')
	#tmp_sen=lines[0]+'\t'+lines[1]
	fw.write(lines[0]+'\t'+lines[1]+'\t'+lines[-2]+'\t'+lines[3]+'\t'+lines[-3]+'\t'+lines[4]+'\n')
	'''
	if(tmp_sen!=lines[0]+'\t'+lines[1]):
		fw.write(lines[0]+'\t'+lines[1]+'\t'+lines[-2]+'\t'+lines[3]+'\t'+lines[-3]+'\t'+lines[4]+'\n')	
		already_sen[tmp_sen]=1
		num+=1
	tmp=lines[0]
	tmp_sen=lines[0]+'\t'+lines[1]
	'''
print num
