# -*- encoding = gb18030 -*-
import math
from abc import ABCMeta, abstractmethod

from deep.dataloader.util import *



class CorpusReader:
    """
    Every CorpusReader should implement this abstract class.
    """
    __metaclass__ = ABCMeta
    train_set = None
    valid_set = None
    test_set = None
    docs = None
    stopwords = None
    word2index = None
    special_flag = set(['<EMPTY>', '<BEG>', '<END>', '<UNK>'])
    
    def __init__(self, dataset_file=None, stopwords_file=None,
                 dict_file=None, word_embedding_file=None,
                 train_valid_test_rate=[0.999, 0.0003, 0.0007], charset='utf8',
                 is_BEG_available=True, is_END_available=True):
        
        if not is_BEG_available:
            self.special_flag.remove('<BEG>')
        
        if not is_END_available:
            self.special_flag.remove('<END>')
        # load stop words
        print ('Reading stop words')
        self.stopwords = load_stopwords_file(stopwords_file, charset)
        print ('Number of stop words: %i' % len(self.stopwords))
        # load word2index
        print ('Reading word2index')
        if dict_file is None:
            raise Exception('Word2index should exist')
        else:
            self.word2index, self.index2word = load_dictionary_file(dict_file,
                                                                    special_flag=self.special_flag, 
                                                                    charset=charset)
            print ('Dictionary size: %i' % len(self.word2index))
        # load word embedding
        if word_embedding_file is not None:
            print ('Reading word embedding.')
            self.word_embedding_dict = load_word_embeddings_file(word_embedding_file, charset)
            print ('Merging word embedding')
            self.word2index, self.index2word, self.word_embedding_matrx = \
                merge_dict_and_embedding(self.word2index, self.special_flag, self.word_embedding_dict)
        # dump word2index used
        print ('Writing word2index used')
        word2index_items = sorted(self.word2index.iteritems(), key=lambda x: x[1])
        with codecs.open(dict_file + '.used', 'w', charset) as fw :
            for word, index in word2index_items:
                fw.write('%s\t%d\n' % (word, index))
        # split dataset into train, valid and test
        if dataset_file is not None:
            print ('loading data')
            dataset = self.load_data(dataset_file, charset)
            print ('dividing data set')
            if(len(dataset[0])==2):
                self.train_set, self.valid_set, self.test_set = self.divide_data_set(dataset, train_valid_test_rate)
            if(len(dataset[0])==3):
                self.train_set, self.valid_set, self.test_set = self.divide_data_set_topic(dataset, train_valid_test_rate)
            print ('Train_set size: %i' % len(self.train_set[0]))
            print ('Valid_set size: %i' % len(self.valid_set[0]))
            print ('Test_set size: %i' % len(self.test_set[0]))
            
        print ('Finish reading data')
            
        
    def get_word_dictionary(self):
        return self.word2index
    
    
    def get_embedding_matrix(self):
        return numpy.matrix(self.word_embedding_matrx, dtype=config.globalFloatType())
    
    
    def get_embedding_matrix_without_special_flag(self):
        return numpy.matrix(self.word_embedding_matrx[len(self.special_flag):], dtype=config.globalFloatType()), self.special_flag
    
    
    def get_train_set(self, scope=None):
        return self.get_model_input(scope, self.train_set[0], self.train_set[1])
        
        
    def get_valid_set(self, scope=None):
        return self.get_model_input(scope, self.valid_set[0], self.valid_set[1])
        
        
    def get_test_set(self, scope=None):
        return self.get_model_input(scope, self.test_set[0], self.test_set[1])
    
    
    def get_y_dimension(self):
        return len(self.train_set[1][0])
    
    
    def get_size(self):
        """
        :return len(self.train_set), len(self.valid_set), len(self.test_set)
        """
        return len(self.train_set[0]), len(self.valid_set[0]), len(self.test_set[0])
    
    
    @abstractmethod
    def load_data(self, dataset_file, charset='utf8'):
        pass
    
    
    @abstractmethod
    def get_model_input(self, scope, x, y):
        """
        :return transformed x, transformed y, original x, original y
        """
        pass
    
    
    @abstractmethod
    def shuffle(self):
        pass
 
 
    def divide_data_post_process(self, train_set, valid_set, test_set):
        """
         This method is not abstract, however, the derived classes may cover this method.
        """
        pass


    def divide_data_set(self, doc_list, train_valid_test_rate):
        n_train = int(math.floor(train_valid_test_rate[0] * len(doc_list)))
        n_train_valid = int(math.floor((train_valid_test_rate[0] +
                                        train_valid_test_rate[1]) * len(doc_list)))
        n_train_valid_test = int(math.floor((train_valid_test_rate[0] +
                                             train_valid_test_rate[1] +
                                             train_valid_test_rate[2]) * len(doc_list)))
        train_set = doc_list[0:n_train]
        valid_set = doc_list[n_train:n_train_valid]
        test_set = doc_list[n_train_valid:n_train_valid_test]    
        train_set_q, train_set_a = zip(*train_set)
        valid_set_q, valid_set_a = zip(*valid_set)
        test_set_q, test_set_a = zip(*test_set)
        
        return (train_set_q, train_set_a), (valid_set_q, valid_set_a), (test_set_q, test_set_a)

    def divide_data_set_topic(self, doc_list, train_valid_test_rate):
        n_train = int(math.floor(train_valid_test_rate[0] * len(doc_list)))
        n_train_valid = int(math.floor((train_valid_test_rate[0] +
                                        train_valid_test_rate[1]) * len(doc_list)))
        n_train_valid_test = int(math.floor((train_valid_test_rate[0] +
                                             train_valid_test_rate[1] +
                                             train_valid_test_rate[2]) * len(doc_list)))
        train_set = doc_list[0:n_train]
        valid_set = doc_list[n_train:n_train_valid]
        test_set = doc_list[n_train_valid:n_train_valid_test]
        train_set_q, train_set_a,train_set_t = zip(*train_set)
        valid_set_q, valid_set_a,valid_set_t = zip(*valid_set)
        test_set_q, test_set_a,test_set_t = zip(*test_set)

        return (train_set_q, train_set_a,train_set_t), (valid_set_q, valid_set_a,valid_set_t), (test_set_q, test_set_a,test_set_t)
 
    def transform_input_data(self, sentence):
        if isinstance(sentence, list) or isinstance(sentence, tuple):
            s_list = list()
            for s in sentence:
                s_list.append(load_sentence(s, self.word2index, special_flag=self.special_flag))
            x_data, x_mask = get_mask_data(s_list)
            return x_data, x_mask
        else:
            sentence = load_sentence(sentence, self.word2index, special_flag=self.special_flag)
            x_data = numpy.transpose(numpy.asmatrix(sentence, 'int64'))
            x_mask = numpy.ones((len(sentence), 1)).astype(config.globalFloatType())
            x_mask = numpy.asmatrix(x_mask)
            return (x_data, x_mask)
    
    
    def transform_input_text(self, sentenceIndecies):
        l = [self.index2word[s] for s in sentenceIndecies]
        return l
