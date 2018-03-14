import codecs
import string
import numpy

import deep.util.config as config



def load_stopwords_file(filename, charset='utf8'):
    stopwords_set = set()
    with codecs.open(filename, 'r', charset) as fo: 
        for line in fo.readlines() :
            stopwords_set.add(line.strip())
            
    return stopwords_set


def load_dictionary_file(filename, special_flag, charset='utf8'):
    word2index = dict()
    index2word = dict()
    
    index2word[len(index2word)] = '<EMPTY>'
    word2index['<EMPTY>'] = len(word2index)
    index2word[len(index2word)] = '<UNK>'
    word2index['<UNK>'] = len(word2index)
    if '<BEG>' in special_flag :
        index2word[len(index2word)] = '<BEG>'
        word2index['<BEG>'] = len(word2index)
    if '<END>' in special_flag :
        index2word[len(index2word)] = '<END>'
        word2index['<END>'] = len(word2index)
        
    with codecs.open(filename, 'r', charset) as fo :
        for line in fo.readlines() :
            data = line.strip().split('\t')
            if len(data) < 1:
                continue
            # process each word
            word = data[0].strip()
            if word is None or len(word) == 0:
                print ('Get a empty line in dict file, continue.\n')
                continue
            if word == '</s>':
                print ('Get </s> in dict file, continue and process later.\n')
                continue
            index = len(word2index)
            word2index[word] = index
            index2word[index] = word
    
    return word2index, index2word


def load_word_embeddings_file(filename, charset='utf-8'):
    
    with codecs.open(filename, 'r', charset) as fo:
        word_embedding = dict()
        last_size = -1
        for line in fo.readlines() :
            data = line.strip().split(' ')
            word = data[0]
            if word == '</s>':
                word = '<END>'
            vec = map(lambda s: string.atof(s), data[1:]);
            if last_size != -1 and len(vec) != last_size:
                print ('A line is shorter than others, skip it.')
                continue
            last_size = len(vec)
            word_embedding[word] = vec
    return word_embedding
  
  
def load_sentence(sentence_str, word2index, special_flag,
                  min_sentence_word_num=0, max_sentence_word_num=(2 << 31) - 1,
                  stopwords=None):
    sentence_str = sentence_str.strip()
    sentence = sentence_str.split(' ')
    n_tokens = len(sentence)
    if n_tokens < min_sentence_word_num or n_tokens > max_sentence_word_num :
        return None
    if stopwords :
        sentence = map(lambda word: word2index[word.strip()] \
                       if (word.strip() in word2index and word.strip() not in stopwords) 
                       else word2index['<UNK>'], sentence)
    else:
        sentence = map(lambda word: word2index[word.strip()] \
                       if (word.strip() in word2index) 
                       else word2index['<UNK>'], sentence)
#     sentence = filter(lambda s: s != word2index['<UNK>'], sentence)
    n_tokens = len(sentence)
    if n_tokens < min_sentence_word_num :
        return None
    if '<BEG>' in special_flag :
        sentence = [word2index['<BEG>']] + sentence
    if '<END>' in special_flag :
        sentence = sentence + [word2index['<END>']]
        
    return sentence


def get_mask_data(batch):
    n_samples = len(batch)
    lengths = [len(s) for s in batch]
    maxlen = numpy.max(lengths)
    data = numpy.zeros((maxlen, n_samples)).astype('int64')
    mask = numpy.zeros((maxlen, n_samples)).astype(config.globalFloatType())
    for idx, s in enumerate(batch):
        data[:lengths[idx], idx] = s
        mask[:lengths[idx], idx] = 1.
        
    return data, mask

def get_mask_data_topic(batch):
    n_samples = len(batch)
    lengths = [len(s) for s in batch]
    maxlen = numpy.max(lengths)
    data = numpy.zeros((maxlen, n_samples)).astype(config.globalFloatType())
    mask = numpy.zeros((maxlen, n_samples)).astype(config.globalFloatType())
    for idx, s in enumerate(batch):
        data[:lengths[idx], idx] = s
        mask[:lengths[idx], idx] = 1.
        
    return data, mask
    
def merge_dict_and_embedding(word_index_dict, special_flag, word_embedding_dict):
    """
    Get the intersect of the dict and word embedding vectors.
    """
    inter_set = set(word_embedding_dict.keys()) & set(word_index_dict.keys())
    word2index = dict()
    index2word = dict()
    embedding_matrx = list()
    zeros = [0] * len(word_embedding_dict.values()[0])
    
    def store_special_token(token):
        if token in special_flag:
            index2word[len(word2index)] = token
            word2index[token] = len(word2index)
            if not token in  word_embedding_dict:
                embedding_matrx.append(zeros)
            else:
                embedding_matrx.append(word_embedding_dict[token])
                
    store_special_token('<EMPTY>')
    store_special_token('<UNK>')
    store_special_token('<BEG>')
    store_special_token('<END>')
    for word in inter_set:
        if word in word2index or word in special_flag:
            continue
        index2word[len(word2index)] = word
        word2index[word] = len(word2index)
        embedding_matrx.append(word_embedding_dict[word])
    
    return word2index, index2word, embedding_matrx
