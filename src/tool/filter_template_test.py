import sys
main_function=sys.argv[2]
f=open(sys.argv[1]+'.data.'+main_function,'r')
fw=open(sys.argv[1]+'.template.'+main_function,'w')
tmp=''
for line in f:
	lines=line.strip().split('\t')
	if(tmp!=lines[1]):
		fw.write(line)
		if(False):
			#pass
			fw.write(lines[1]+'\t'+'\t'.join(lines[1:])+'\n')
	tmp=lines[1]
f.close()
fw.close()
