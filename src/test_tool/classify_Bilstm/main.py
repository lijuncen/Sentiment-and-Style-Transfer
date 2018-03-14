# -*- encoding = gb18030 -*-
"""
This file are the entrance of all the dialog processors.
Notice that every processors should implement algorithm interface.

example of args:
[dir, dialog_path, dict_path, stopwords_path, embedding_path, train_rate, valid_rate, test_rate, algo_name, method]
e.g.
plp plp/dialog  plp/dict punct plp.word.vec 0.9999 0.00007 0.00003 skip_thought train
test_abc test_abc/dialogs.10  test_abc/dict.8k test_abc/stopwords.txt  test_abc/embeddings.txt 0.5 0.25 0.25 ChoEncoderDecoder train
"""
import os
import sys
import string
import logging

import deep.util.config as config 



if __name__ == '__main__':
    #sys.argv = ['dialog.py', 'test2', 'test/dialog.txt', 'test/dict.txt', 'punct.txt', 'embedings.txt', '0.3', '0.3', '0.3', 'ChoEncoderDecoderTopic', 'train']
    #sys.argv = ['dialog.py', 'test2', 'test/dialog.txt', 'test/dict.txt', 'punct.txt', 'embedings.txt', '0.3', '0.3', '0.3', 'ChoEncoderDecoderTopic', 'generate_b_v_t']
    base_path = os.path.join(os.getcwd(), 'data')
    dataset_folder =sys.argv[1]
    dataset_file=sys.argv[2]
    dict_file=sys.argv[3]
    stopwords_file =sys.argv[4]
    word_embedding_file=sys.argv[5]
    '''
    dataset_folder = os.path.join(base_path, sys.argv[1]) 
    dataset_file = os.path.join(base_path, sys.argv[2])
    dict_file = os.path.join(base_path, sys.argv[3])
    stopwords_file = os.path.join(base_path, sys.argv[4])
    word_embedding_file = os.path.join(base_path, sys.argv[5])
    '''
    train_rate = string.atof(sys.argv[6])
    valid_rate = string.atof(sys.argv[7])
    test_rate = string.atof(sys.argv[8])
    algo_name = sys.argv[9]
    mode = sys.argv[10]
    charset = config.globalCharSet()
    print ('dataset file: %s.' % (dataset_file))
    print ('dict file: %s.' % (dict_file))
    print ('stopwords file: %s.' % (stopwords_file))
    print ('word embedding file: %s.' % (word_embedding_file))
    print ('algorithms name: %s.' % (algo_name))
    print ('mode: %s.' % (mode))
    print ('charset: %s.' % (charset))
        
    if algo_name == 'SeqToSeq' :
        from deep.manage.model.seq_to_seq import RnnEncoderDecoder
        manager = RnnEncoderDecoder(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                    train_rate, valid_rate, test_rate, algo_name, charset, mode) 
    elif algo_name == 'ChoEncoderDecoder' :
        from deep.manage.model.cho_encoder_decoder import RnnEncoderDecoder
        manager = RnnEncoderDecoder(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                    train_rate, valid_rate, test_rate, algo_name, charset, mode)
    elif algo_name == 'ChoEncoderDecoderTopic' :
        from deep.manage.model.cho_encoder_decoder_topic import RnnEncoderDecoder
        manager = RnnEncoderDecoder(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                    train_rate, valid_rate, test_rate, algo_name, charset, mode)
    elif algo_name == 'ChoEncoderDecoderDT' :
        from deep.manage.model.cho_encoder_decoder_DT import RnnEncoderDecoder
        manager = RnnEncoderDecoder(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                    train_rate, valid_rate, test_rate, algo_name, charset, mode)
    elif algo_name == 'TegEncoderDecoder' :
        from deep.manage.model.teg_encoder_decoder import RnnEncoderDecoder
        manager = RnnEncoderDecoder(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                    train_rate, valid_rate, test_rate, algo_name, charset, mode)
    elif algo_name == 'BiEncoderAttentionDecoder' :
        from deep.manage.model.biencoder_attention_decoder import BiEncoderAttentionDecoder
        manager = BiEncoderAttentionDecoder(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                            train_rate, valid_rate, test_rate, algo_name, charset, mode)
    elif algo_name == 'BiEncoderAttentionDecoderStyle' :
        from deep.manage.model.biencoder_attention_decoder_style import BiEncoderAttentionDecoderStyle
        manager = BiEncoderAttentionDecoderStyle(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                            train_rate, valid_rate, test_rate, algo_name, charset, mode)    
    elif algo_name == 'LihangEncoderDecoder' :
        from deep.manage.model.lihang_encoder_decoder import LihangEncoderDecoder
        manager = LihangEncoderDecoder(dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                                       train_rate, valid_rate, test_rate, algo_name, charset, mode)
    if mode == 'train':
        manager.train()
    elif mode == 'generate':
        input_file = os.path.join(os.getcwd(), 'measure', 'generate_test')
        output_file = os.path.join(os.getcwd(), 'measure', 'generate_test_res')
        manager.generate(input_file=input_file, output_file=output_file)
    elif mode == 'generate_b_v':
        #input_file = os.path.join(base_path, 'measure', 'generate_test_word')
        #output_file = os.path.join(base_path, 'measure', 'generate_test_word_res')
        input_file = './data/create_IR_data/ir_data.orgin'
        output_file= './data/create_IR_data/ir_data.orgin.cost'
        manager.generate_b_v(input_file=input_file, output_file=output_file)
    elif mode == 'generate_b_v_t':
        #input_file = os.path.join(base_path, 'measure', 'gen_test.txt')
        input_file=sys.argv[11]
        #input_file='instance_test.txt.gb18030'
        #output_file = os.path.join(base_path, 'measure', 'generate_test_zi_res')
        #output_file='generate_test_zi_topic_res_new'
        output_file=sys.argv[11]+'.result'
        print output_file
        manager.generate_b_v_t(input_file=input_file, output_file=output_file)
    elif mode == 'generate_b_v_t_c':
        #input_file = os.path.join(base_path, 'measure', 'gen_test.txt')
        input_file=sys.argv[11]
        #input_file='instance_test.txt.gb18030'
        #output_file = os.path.join(base_path, 'measure', 'generate_test_zi_res')
        #output_file='generate_test_zi_topic_res_new'
        output_file=sys.argv[11]+'.result'
        print output_file
        manager.generate_b_v_t_c(input_file=input_file, output_file=output_file)
    elif mode == 'generate_b_v_t_g':
        #input_file = os.path.join(base_path, 'measure', 'gen_test.txt')
        input_file='./data/chat_pair_new/DT_test_data.txt.200'
        #input_file='instance_test.txt.gb18030'
        #output_file = os.path.join(base_path, 'measure', 'generate_test_zi_res')
        #output_file='generate_test_zi_topic_res_new'
        output_file=input_file+'.gen_result'
        print output_file
        manager.generate_b_v_t_g(input_file=input_file, output_file=output_file)
    elif mode == 'observe':
        manager.observe()
            
    print ('All finished!')
