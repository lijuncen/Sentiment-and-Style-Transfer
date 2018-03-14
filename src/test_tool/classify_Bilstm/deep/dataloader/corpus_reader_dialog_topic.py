# -*- encoding = gb18030 -*-
import os 

from multiprocessing.dummy import Pool as ThreadPool

from deep.dataloader.util import *
from deep.dataloader.corpus_reader import CorpusReader 

    
    
class CorpusReaderDialogTopic(CorpusReader):
    
    def __init__(self, dataset_file=None, min_dialog_sentence_num=2, \
                 max_sentence_word_num=1 << 31, min_sentence_word_num=1, \
                 stopwords_file=None, dict_file=None, \
                 word_embedding_file=None, \
                 train_valid_test_rate=[0.999, 0.0003, 0.0007], charset='utf-8',
                 is_BEG_available=True, is_END_available=True):
        self.min_dialog_sentence_num = min_dialog_sentence_num
        self.max_sentence_word_num = max_sentence_word_num
        self.min_sentence_word_num = min_sentence_word_num
        
        CorpusReader.__init__(self, dataset_file=dataset_file, stopwords_file=stopwords_file, \
                              dict_file=dict_file, word_embedding_file=word_embedding_file, charset=charset, \
                              train_valid_test_rate=train_valid_test_rate,
                              is_BEG_available=is_BEG_available,
                              is_END_available=is_END_available)
    
        measure_base_dir = os.path.split(os.path.realpath(__file__))[0] + '/../../measure'
        #measure_base_dir='E:\\new_proj\\DeepEmbedding-master\\DeepEmbedding\\measure'
        self.pr_set = self.load_data(measure_base_dir + '/PR_test_zi.topic', charset)
        self.pr_set = zip(*self.pr_set)
        #self.whole_set=self.load_data('./data/style_transfer/funny_all.txt.out1.gbkchanged', charset)
        self.whole_set=[]
        self.answer_set=[i[1][:] for i in self.whole_set]


    def get_answer_set(self):
        #print self.train_set
        return self.answer_set
        
    def get_pr_set(self, scope=None, shuffle=False, merge=False):
        return self.get_model_input(None, self.pr_set[0], self.pr_set[1],self.pr_set[2], \
                                   shuffle=shuffle, merge=merge)
        
    def shuffle(self):
        pass
    
    
    def get_train_set(self, scope=None, shuffle=False, merge=False):
        return self.get_model_input(scope, self.train_set[0], self.train_set[1], self.train_set[2],\
                                   shuffle=shuffle, merge=merge)
        
        
    def get_valid_set(self, scope=None, shuffle=False, merge=False):
        return self.get_model_input(scope, self.valid_set[0], self.valid_set[1], self.valid_set[2],\
                                   shuffle=shuffle, merge=merge)
        
        
    def get_test_set(self, scope=None, shuffle=False, merge=False):
        return self.get_model_input(scope, self.test_set[0], self.test_set[1], self.test_set[2],\
                                   shuffle=shuffle, merge=merge)
    
    
    def get_model_input(self, scope, question, answer, topic,shuffle=False, merge=False):
        if scope is not None:
            scope = list(scope)
            scope[1] = numpy.min([scope[1], len(question)])
            if(scope[0] < 0 or scope[0] >= scope[1]):
                return None
        else:
            scope = [0, len(question)]
        batch_question = question[scope[0]:scope[1]]
        batch_answer = answer[scope[0]:scope[1]]
        batch_topic= topic[scope[0]:scope[1]]
        if shuffle:
            idx = range(len(batch_answer))
            numpy.random.shuffle(idx)
            batch_answer = [batch_answer[i] for i in idx]
        if not merge:
            question_data, question_mask = get_mask_data(batch_question)
            answer_data, answer_mask = get_mask_data(batch_answer)
            topic_data,topic_mask = get_mask_data_topic(batch_topic)
            return (question_data, question_mask), (answer_data, answer_mask),(topic_data,topic_mask), batch_question, batch_answer
        else:
            batch = [question + answer +topic for (question, answer,topic) in zip(batch_question, batch_answer,batch_topic)]
            data, mask = get_mask_data(batch)
            return (data, mask), None, batch, None
    

    def load_data(self, dataset_file, charset='utf-8'):
        pool = ThreadPool(12)
        
        def deal_one_sentence(lines):
            dialogs = list()
            for line in lines:
                line = line.strip()
                sentences = line.split('\t')
                if len(sentences) < self.min_dialog_sentence_num:
                    continue
                dialog = map(lambda sentence:load_sentence(sentence, self.word2index,
                                                          special_flag=self.special_flag,
                                                          min_sentence_word_num=self.min_sentence_word_num,
                                                          max_sentence_word_num=self.max_sentence_word_num,
                                                          stopwords=self.stopwords),
                             sentences)
                if None in dialog or len(dialog) < 2: 
                    continue
                dialog=dialog[:2]
                # dialog is a list [q1, a1, q2, a2, ...]
                if(len(sentences)==3):
                    dialogs.extend(zip(dialog[:-1], dialog[1:],[[string.atof(sentences[2])]]))
            dialogs = [[q[:-1], [q[-1]] + a, t ]for q, a,t in dialogs]
            
            return dialogs
        
        print ('loading data into memory')
        lines = list()
        doc_list = list()
        with codecs.open(dataset_file, 'r', charset) as fo:
            for line in fo.readlines() :
                lines.append(line)
        print ('Processing data')
        total_data_size = len(lines)
        block_number = 24
        block_size = (total_data_size - 1) / block_number + 1
        lines = [lines[i * block_size:(i + 1) * block_size] for i in range(block_number)]
        lines = map(deal_one_sentence, lines)
        pool.close()
        print ('Insert into doc list')
        for doc in lines:
            if doc is not None:
                doc_list.extend(doc)
                
        return doc_list


    def divide_data_post_process(self, train_set, valid_set, test_set):
        
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: max(len(seq[x][0]), len(seq[x][1])))
        
        train_set = [train_set[i] for i in len_argsort(train_set)]
        valid_set = [valid_set[i] for i in len_argsort(valid_set)]
        test_set = [test_set[i] for i in len_argsort(test_set)]
        
        return train_set, valid_set, test_set
    
