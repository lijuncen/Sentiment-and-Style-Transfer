import sys
import random
import string
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


f=open(sys.argv[1]+'.tf_idf.TemplateBased','r')
words_dict=[]
word_dict={}
num=0
for line in f:
	lines=line.strip().split('\t')
	if(len(lines)<2):
		continue
	if(string.atof(lines[1])>string.atof(sys.argv[4]) and num<string.atof(sys.argv[5])):	
		word_dict[lines[0]]=1
	num+=1
f.close()
print sys.argv[4]
print sys.argv[5]
print len(word_dict.keys())
for i in word_dict.keys():
	words=i.split(' ')
	words_dict.append(words)
#print len(word_dict)
f=open(sys.argv[2],'r')
fw=open(sys.argv[3]+'.template1','w')
total_num=0
change_num=0
for line in f:
	lines=line.strip().split(' ')
	content=''
	style=''
	style_dict=[]
	for i in range(len(lines)):
		for n in range(4,0,-1):
			if(i+n>len(lines)):
				continue
			if(word_dict.get(' '.join(lines[i:i+n]))!=None and (style_dict==[] or i+n-1 >style_dict[-1])):
				style+=' '.join(lines[i:i+n])+' '
				style_dict.append(i)
				style_dict.append(i+n-1)
				break
	style_dict_merge=[]
	i=0
	while(i<len(style_dict)):
		start=style_dict[i]
		end=style_dict[i+1]
		n=2
		while(i+n<len(style_dict)):
			if(style_dict[i+n]<=end):
				if(style_dict[i+n+1]>end):
					end=style_dict[i+n+1]
			elif(style_dict[i+n]==end+1):
				end=style_dict[i+n+1]
			else:
				#style_dict_merge.append(start)
				#style_dict_merge.append(end)
				#i=i+n
				break
			n+=2
		style_dict_merge.append(start)
		style_dict_merge.append(end)
		i=i+n	
	#print style_dict,style_dict_merge
	start=0
	style1=''
	for i in range(0,len(style_dict_merge),2):
		style1+=' '.join(lines[style_dict_merge[i]:style_dict_merge[i+1]+1])+'######'
	#print line
	#print style
	#print style_dict
	if(len(style_dict)>0 and style_dict[0]==0):
		content='slotholdplace '
	for i in range(0,len(style_dict),2):
		if(start<style_dict[i]):
			content+=' '.join(lines[start:style_dict[i]])+' '
			if(1):
				content+='slotholdplace '
		start=style_dict[i+1]+1
	if(start<len(lines)):
		content+=' '.join(lines[start:len(lines)])+' '
	#print content+'\n'
	style=style.strip()
	content=content.strip()
	contents=content.strip().split(' ')
	if(len(contents)<5 and 'train' in sys.argv[2]):
		continue
	total_num+=1
	if style!='':
		fw.write(content+'\t'+style1[:-6]+'\n')
	else:
		if('train' not in sys.argv[2]):
			fw.write(content+'\t'+'SELF'+'\n')
f.close()
fw.close()
print change_num
print total_num
