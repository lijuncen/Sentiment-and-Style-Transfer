import sys
f=open(sys.argv[1],'r')
fw=open(sys.argv[2]+'.lm','w')
for line in f:
	fw.write('1\t'+line.strip()+'\t1\t1\n')
f.close()
fw.close()
