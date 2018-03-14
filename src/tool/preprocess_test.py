import sys
import random
import string
operation=sys.argv[3]
dict_num=string.atof(sys.argv[4])
dict_thre=string.atof(sys.argv[5])

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


f=open(sys.argv[2]+'.tf_idf'+'.'+operation,'r')
words_dict=[]
word_dict={}
num=0
for line in f:
	try:
		lines=line.strip().decode('utf-8').encode('gb18030').split('\t')
	except:
		continue
	#print len(lines)
	#print line
	if(len(lines)!=2):
		continue 
	if(string.atof(lines[1])>dict_thre and num<dict_num):	
		word_dict[lines[0]]=1
		num+=1
f.close()
for i in word_dict.keys():
	words=i.split(' ')
	words_dict.append(words)
#print len(word_dict)
f=open(sys.argv[1],'r')
fw=open(sys.argv[6]+'.data.'+operation,'w')
total_num=0
change_num=0
for line in f:
	try:
		lines=line.strip().decode('utf-8').encode('gb18030').split(' ')
	except:
		print line
		continue
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
	start=0
	#print line
	#print style
	#print style_dict
	if(len(style_dict)>0 and style_dict[0]==0):
		content=''
	for i in range(0,len(style_dict),2):
		if(start<style_dict[i]):
			content+=' '.join(lines[start:style_dict[i]])+' '
			if(1):
				pass
		start=style_dict[i+1]+1
	if(start<len(lines)):
		content+=' '.join(lines[start:len(lines)])+' '
	#print content+'\n'
	style=style.strip()
	content=content.strip()
	contents=content.strip().split(' ')
	if(content==''):
		content='EMPTY'
	if(len(contents)<5):
		pass
	total_num+=1
	if style!='':
		if operation=='label':
			style=sys.argv[1][-1]
			fw.write(content+'\t'+line.strip()+'\t'+style+'\t'+'1'+'\n')
			continue
		#fw.write(content+'\t'+line.strip()+'\t'+style+'\t'+'1'+'\n')
		if(operation=='entire'):
			fw.write(line.strip()+'\t'+line.strip()+'\t'+style+'\t'+'1'+'\n')
			continue
		fw.write(content+'\t'+line.strip()+'\t'+style+'\t'+'1'+'\n')
		alreay_style={}
		alreay_style[style]=1
		for hh in range(0):
			style_tmp=''
			random.shuffle(words_dict)
			word_dict_array=words_dict[:1000]
			for i in range(0,len(style_dict),2):
				style_word_array=[lines[x] for x in range(style_dict[i],style_dict[i+1]+1)]
				#print style_word_array
				#print style
				flag=1
				style_i_tmp=''
				for word_dict_i in word_dict_array:
					ed=levenshteinDistance(word_dict_i,style_word_array)
					if(((len(style_word_array)>=2 or (len(style_word_array)==1 and len(word_dict_i)==2)) and ed ==1)):
						style_i_tmp=' '.join(word_dict_i)
						flag=0
						break
				if(flag==1):
					style_i_tmp=' '.join(style_word_array)
				style_tmp+=style_i_tmp+' '
			if(alreay_style.get(style_tmp)==None):
				fw.write(content+'\t'+line.strip()+'\t'+style_tmp+'\t'+'1'+'\n')
				alreay_style[style_tmp]=1
				#print style_tmp
				#print style	
		change_num+=1
	else:
		fw.write(content+'\t'+line.strip()+'\t'+'self'+'\t'+'1'+'\n')
f.close()
fw.close()
print change_num
print total_num
