# -*- encoding = gb18030 -*-
import os

from deep.manage.manager import ModelManager
from deep.util.parameter_operation import load_params_val, get_params_file_name, save_confs_val
from deep.algorithms.networks.biencoder_attention_decoder_style import BiEncoderAttentionDecoderStyleNetwork
from deep.dataloader.corpus_reader_dialog import CorpusReaderDialog



class BiEncoderAttentionDecoderStyle(ModelManager) :
    
    def __init__(self, dataset_folder, dataset_file, dict_file, stopwords_file, word_embedding_file,
                 train_rate, valid_rate, test_rate, algo_name, charset, mode) :
        """
        Need to set these attributes.
            1. conf_dict: configuration of the model.
            2. cr: CorpursReader for operate data.
            3. model: the network model.
        """
        self.conf_dict = {'algo_name': algo_name, 'batch_size': 2,
                          'train_valid_test_rate': [train_rate, valid_rate, test_rate],
                          'split_level': 'zi', 'pre_word_embedding': False,
                          'word_embedding_dim': 128, 'max_sentence_word_num': 150,
                          'min_sentence_word_num': 1, 'is_BEG': False, 'is_END': True,
                          'hidden_dim': 1024, 'style_number': 64,
                          'style_dim': 128, 'charset': charset, 'shuffle': False,
                          'save_freq': 100}
        if mode == 'train' :
            self.param_path = \
                os.path.join(dataset_folder, 'model', 'dialog', get_params_file_name(self.conf_dict) + '.model')
        else :
            self.param_path = \
                os.path.join(dataset_folder, 'model', 'dialog', get_params_file_name(self.conf_dict) + '.model.final')
        param_dict = load_params_val(self.param_path)
        self.conf_path = os.path.join(dataset_folder, 'model', 'dialog', get_params_file_name(self.conf_dict) + '.conf')
        save_confs_val(self.conf_dict, self.conf_path)
        # set corpus reader
        if mode == 'train' :
            self.cr = CorpusReaderDialog(dataset_file=dataset_file,
                                         stopwords_file=stopwords_file,
                                         dict_file=dict_file,
                                         word_embedding_file=None,
                                         train_valid_test_rate=self.conf_dict['train_valid_test_rate'],
                                         charset=self.conf_dict['charset'],
                                         max_sentence_word_num=self.conf_dict['max_sentence_word_num'],
                                         min_sentence_word_num=self.conf_dict['min_sentence_word_num'],
                                         is_BEG_available=self.conf_dict['is_BEG'],
                                         is_END_available=self.conf_dict['is_END'])
        else :
            self.cr = CorpusReaderDialog(dataset_file=None,
                                         stopwords_file=stopwords_file,
                                         dict_file=dict_file,
                                         word_embedding_file=None,
                                         train_valid_test_rate=self.conf_dict['train_valid_test_rate'],
                                         charset=self.conf_dict['charset'],
                                         max_sentence_word_num=self.conf_dict['max_sentence_word_num'],
                                         min_sentence_word_num=self.conf_dict['min_sentence_word_num'],
                                         is_BEG_available=self.conf_dict['is_BEG'],
                                         is_END_available=self.conf_dict['is_END'])
        # set model
        self.model = BiEncoderAttentionDecoderStyleNetwork(n_words=len(self.cr.get_word_dictionary()),
                                                      hidden_status_dim=self.conf_dict['hidden_dim'],
                                                      word_embedding_dim=self.conf_dict['word_embedding_dim'],
                                                      style_number=self.conf_dict['style_number'],
                                                      style_dim=self.conf_dict['style_dim'],
                                                      input_params=param_dict)
