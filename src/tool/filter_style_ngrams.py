import sys
import random
import string

data_prefix=sys.argv[1]
result_prefix=sys.argv[4]
data_file_num=string.atoi(sys.argv[2])
negtive_array=[]
def load_data(file_name):
	f=open(file_name,'r')
	tmp=[]
	for line in f:
		if('url' in line):
			continue
		line=line.strip()
		tmp.append(line)
	random.shuffle(tmp)
	return tmp[:min(len(tmp),5000000000)]
#negtive_array+=load_data('all_subtitle.gbk')
#negtive_array+=load_data('all_daily_dialogtwitter_sen.txt')
name_array=[]
for i in range(0,data_file_num):
	name_array.append(data_prefix+str(i))
#name_array=['funny_train.txt.test','romantic_train.txt.test']
#tag='Sheldon'
def get_dict(sen_array):
        tmp_dict={}
        for i in sen_array:
                sens=i.strip().split(' ')
                for n in range(1,5):
                        for l in range(0,len(sens)-n+1):
                                tmp=' '.join(sens[l:l+n])
                                if(tmp_dict.get(tmp)!=None):
                                        tmp_dict[tmp]+=1
                                else:
                                        tmp_dict[tmp]=1
        return tmp_dict
num=0
for tag in name_array:
	negtive_array=[]
	positve_array=load_data(tag)
	for i in name_array:
		if tag!=i:
			negtive_array+=load_data(i)
	#fw=open(tag+'.train','w')
	#neg_dict={}
	#pos_dict={}
	neg_dict=get_dict(negtive_array)
	pos_dict=get_dict(positve_array)
	tf_idf={}
	for i in pos_dict.keys():
        	if(neg_dict.get(i)!=None):
                	tf_idf[i]=(pos_dict[i]+1.0)/(neg_dict[i]+1.0)
        	else:
                	tf_idf[i]=(pos_dict[i]+1.0)
	tf_dif1=sorted(tf_idf.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
	fw=open(result_prefix+str(num)+'.tf_idf'+'.'+sys.argv[3],'w')
	for i in tf_dif1:
        	fw.write(i[0]+'\t'+str(i[1])+'\n')
	num+=1
