import sys
import os
from whoosh.fields import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, And, Phrase, Or
from whoosh import scoring
from whoosh.collectors import TimeLimitCollector, TimeLimit
def process(array):
	if(len(array)<2):
                return None,None,None
        for i in array:
                word=i.split(' ')
                if(len(word)>=50 or len(word)<=0):
                        print len(word)
                        return None,None,None
	context=''
	post=array[0]
	response=array[-1]
	
	
        return context,response,post
def get_cpr(line):
	lines=line.lower().strip().split('\t')
	context=''
	post=lines[0]
	response=lines[1]
	return context.strip(),response,post


def load_train_data(file_name,writer):
	array=[]
	f=open(file_name)
	for line in f:
        	#lines=line.lower().strip().split('\t')
                context,response,post=get_cpr(line)
		if(context!=''):
			writer.add_document(context=context.decode('gb18030'),response=response.decode('gb18030'),post=post.decode('gb18030'))
                	
        	else:
                	writer.add_document(response=response.decode('gb18030'),post=post.decode('gb18030'))
	#writer.add_document(context=u"Everyone has that",response=u"ok")
	#return writer
	writer.commit()
def get_query(line,ix):
	lines=line.strip().split('\t')
	#context=unicode(' '.join(lines[2:-1]), 'gb18030')
	post=unicode(lines[0],'gb18030')
	#q1=QueryParser("context", ix.schema).parse(context)
	q2=QueryParser("post", ix.schema).parse(post)
	#context=' '.join(lines[2:-1])
	#query =QueryParser("post", ix.schema).parse(post)
	terms = list(q2.all_terms())
	query = Or([Term(*x) for x in terms])
	return query
	context=unicode(context,'gb18030')
	q1=QueryParser("context", ix.schema).parse(context)	
	terms = list(q1.all_terms()) + list(q2.all_terms())
        query = Or([Term(*x) for x in terms])
        return query

if __name__ == '__main__':
	schema = Schema(context=TEXT(stored=True), response = STORED,post=TEXT(stored=True))
	#index_path="indexdir"
	index_path="sen1"
	#ix = create_in("/scr/juncenl/retrieve_data/indexdir", schema)
	#if os.path.exists(index_path):
	if False:
		print 'exist' 
		ix = open_dir(index_path)
	#ix = create_in("indexdir", schema)
	else:
		ix = create_in(index_path, schema)
		writer = ix.writer()
		#writer=load_train_data('twitter.txt',writer)
		#load_train_data('/scr/juncenl/retrieve_data/processed.out.out1.processed.filter.train',writer)
		load_train_data('sentiment.train.0.template1',writer)
	#writer.add_document(context=u"Everyone has that",response=u"ok")
	#writer.commit()
	f=open('sentiment.test.1.template1','r')
	fw=open('sentiment.test.1.template1'+'.result','w')
	with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
		c = searcher.collector(limit=10)
		tlc = TimeLimitCollector(c, timelimit=10.0)
		for line in f:
			#line=line.replace('slotholdplace ','').replace(' slotholdplace','')
			query=get_query(line.replace('slotholdplace ','').replace(' slotholdplace',''),ix)
			#print query
			try:
				#results = searcher.search_with_collector(query,tlc)
				searcher.search_with_collector(query,tlc)
				results = tlc.results()
			#fw.write(line.strip()+'\t'+results[0]+'\n')
			#print results[:]
				lines=line.strip().split('\t')
			#print line
				for i in range (min(len(results),10)):
					fw.write(line.strip()+'\t'+str(results[i]["post"])+'\t'+str(results[i]["response"])+'\n')
					#print line
				'''
				if(len(lines)==3):
					print results[0]
					print line.strip()
				'''
			except :
				print line.strip()
				print results
				#pass
		#query = QueryParser("context", ix.schema).parse("Everyone has that")
		#results = searcher.search(query)
		#print results[:]

