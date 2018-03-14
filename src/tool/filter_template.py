import sys
main_function=sys.argv[2]
f=open(sys.argv[1]+'.data.'+main_function,'r')
fw=open(sys.argv[1]+'.template.'+main_function,'w')
tmp=''
for line in f:
	lines=line.strip().split('\t')
	if(lines[-2]!='self' and tmp!=lines[0]):
		fw.write(line)
	tmp=lines[0]
