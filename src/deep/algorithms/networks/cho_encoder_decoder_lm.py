# -*- encoding = utf-8 -*-
from collections import OrderedDict

import numpy
import theano
import theano.tensor as tensor
import theano.printing as printing
from theano.gof.graph import inputs

import deep.util.config as config
from deep.algorithms.util import numpy_floatX
from deep.algorithms.networks.network import Network
# from algorithms.layers.softmax_layer import SoftmaxLayer
from deep.algorithms.layers.maxout_layer import MaxoutLayer
from deep.algorithms.layers.rnn_encoder_layer import EncoderLayer
from deep.algorithms.layers.rnn_decoder_layer import DecoderLayer_Cho
from deep.algorithms.layers.rnn_decoder_layer import DecoderLayer_Seq2Seq
import string
import theano.tensor as T


class RnnEncoderDecoderNetwork(Network):
    """
    This class will process the dialog pair with a encoder-decoder network.
    It has 2 abilities:
        1. Train the language model.
        2. Model the relationship of Q&A
    """

    def init_global_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        params['Wemb_e'] = (0.1 * randn).astype(config.globalFloatType()) 
        randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        #params['Wemb_e'] = (0.1 * randn).astype(config.globalFloatType())
        #randn = numpy.random.rand(options['topic_embedding_dim'], options['topic_embedding_dim'])/options['topic_embedding_dim']*2
        #params['QTA']=(1.0 * randn).astype(config.globalFloatType())
        #randn = numpy.random.rand(options['n_topics'], options['topic_embedding_dim'])
        #params['Temb'] = (0.1 * randn).astype(config.globalFloatType())
        #params['Temb'] = numpy.dot(params['Qemb'],params['QTA'])
        return params


    def __init__(self, n_words, word_embedding_dim=128, hidden_status_dim=128, n_topics=2, topic_embedding_dim=5,input_params=None):
        self.options = options = {
            'n_words': n_words,
            'word_embedding_dim': word_embedding_dim,
            'hidden_status_dim': hidden_status_dim,
            'n_topics' : n_topics,
            'topic_embedding_dim' :topic_embedding_dim,
            'learning_rate': 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
            'optimizer': self.adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
            }
        # global paramters.
        params = self.init_global_params(options)
        # Theano paramters,
        self.tparams = self.init_tparams(params)
        #print self.tparams['Temb'] 
        #self.answer_emb=T.dot(self.tparams['Qemb'],self.tparams['QTA'])
        # Used for dropout.
        # self.use_noise = theano.shared(numpy_floatX(0.))

        # construct network
        theano.config.compute_test_value = 'off'
        self.question = tensor.matrix('question', dtype='int64')
        self.question_mask = tensor.matrix('question_mask', dtype=config.globalFloatType())
        self.style = tensor.matrix('style', dtype='int64')
        self.style_mask = tensor.matrix('style_mask', dtype=config.globalFloatType())
        self.style_embedding=self.tparams['Wemb_e'][self.style.flatten()].reshape(
            [self.style.shape[0], self.style.shape[1], options['word_embedding_dim']])
        #self.question_mask = tensor.matrix('question_mask', dtype='int64')
        self.topic = tensor.matrix('topic', dtype=config.globalFloatType())
        # self.question.tag.test_value = numpy.array([[10, 2, 0], [5, 9, 2]]) # for debug
        # self.question_mask.tag.test_value = numpy.array([[1, 1, 0], [1, 1, 1]]) # for debug
        self.question_embedding = self.tparams['Wemb_e'][self.question.flatten()].reshape(
            [self.question.shape[0], self.question.shape[1], options['word_embedding_dim']])
        '''
        #   1. encoder layer
        self.encoder_layer = EncoderLayer(word_embedding_dim=options['word_embedding_dim'],
                                          hidden_status_dim=options['hidden_status_dim'],
                                          tparams=self.tparams)
        self.encoder_hidden_status1 = self.encoder_layer.getOutput(inputs=(self.question_embedding, self.question_mask))
        self.encoder_layer_style = EncoderLayer(word_embedding_dim=options['word_embedding_dim'],
                                          hidden_status_dim=options['hidden_status_dim'],
                                          tparams=self.tparams,prefix='encoder_layer_style'
                                          )
        self.encoder_hidden_status_style = self.encoder_layer_style.getOutput(inputs=(self.style_embedding, self.style_mask))
        self.encoder_hidden_status=tensor.concatenate([self.encoder_hidden_status1[-1],self.encoder_hidden_status_style[-1]],axis=1).reshape([1,self.encoder_hidden_status_style.shape[1],2*self.encoder_hidden_status_style.shape[2]])
        '''
        #self.topic_states = self.tparams['Temb'][self.topic.flatten()].reshape([1,self.question.shape[1], options['topic_embedding_dim']])
        #self.topic_change=T.alloc(self.topic_states,self.question.shape[0], self.question.shape[1], options['topic_embedding_dim'])
        #self.encoder_hidden_status = T.concatenate([self.encoder_hidden_status,self.topic_change], axis=2)
        #   2. decoder layer
        self.answer = tensor.matrix('answer', dtype='int64')
        self.answer_mask = tensor.matrix('answer_mask', dtype=config.globalFloatType())
        # self.answer.tag.test_value = numpy.array([[11, 10, 2], [5, 2, 0]]) # for debug
        # self.answer_mask.tag.test_value = numpy.array([[1, 1, 1], [1, 1, 0]]) # for debug
        self.answer_embedding = self.tparams['Wemb_e'][self.answer.flatten()].reshape(
            [self.answer.shape[0], self.answer.shape[1], options['word_embedding_dim']])
        '''
        self.decoder_layer = DecoderLayer_Cho(word_embedding_dim=options['word_embedding_dim'] +2*options['hidden_status_dim'],
                                              hidden_status_dim=options['hidden_status_dim'],
                                              tparams=self.tparams)
        self.decoder_hidden_status = self.decoder_layer.getOutput(inputs=(self.answer_embedding, self.answer_mask,
                                                                          self.encoder_hidden_status))
        '''
        self.decoder_layer1 = DecoderLayer_Seq2Seq(word_embedding_dim=options['word_embedding_dim'],
                                              hidden_status_dim=options['hidden_status_dim'],
                                              tparams=self.tparams,prefix='decoder_layer_lm')
	self.decoder_hidden_status1 = self.decoder_layer1.getOutput(inputs=(self.answer_embedding, self.answer_mask,
                                                                          self.answer_embedding))
	

        #   3. maxout  layer
        self.maxout_layer1 = MaxoutLayer(base_dim=options['word_embedding_dim'],
                                                     refer_dim=options["hidden_status_dim"] + options['word_embedding_dim'],
                                                    tparams=self.tparams,
                                                    prefix="maxout1")
        self.maxout_input1 = tensor.concatenate([self.decoder_hidden_status1[:-1, :, :].
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             options['hidden_status_dim']]),
                                                 self.answer_embedding[:-1, :, :].
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             options['word_embedding_dim']])],
                                                axis=1)
        output_error_vector = self.maxout_layer1.negative_log_likelihood(self.tparams['Wemb_e'],
                                                                    self.maxout_input1,
                                                                   y=self.answer[1:, :].flatten())
        '''
        #   3. maxout  layer
        self.maxout_layer = MaxoutLayer(base_dim=options['word_embedding_dim'],
                                                    refer_dim=3 * options["hidden_status_dim"] + options['word_embedding_dim'],
                                                    tparams=self.tparams,
                                                    prefix="maxout")
        self.maxout_input = tensor.concatenate([self.decoder_hidden_status[:-1, :, :].
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             options['hidden_status_dim']]),
                                                 tensor.alloc(self.encoder_hidden_status[-1, :, :],
                                                              self.answer.shape[0] - 1,
                                                              self.answer.shape[1],
                                                              2*options['hidden_status_dim']).
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             2*options['hidden_status_dim']]),
                                                 self.answer_embedding[:-1, :, :].
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             options['word_embedding_dim']])],
                                                axis=1)
        output_error_vector = self.maxout_layer.negative_log_likelihood(self.tparams['Wemb_e'],
                                                                    self.maxout_input,
                                                                    y=self.answer[1:, :].flatten())
        '''
        self.topic_matrix=tensor.alloc(self.topic.flatten(),self.answer.shape[0] - 1,self.answer.shape[1]).flatten()
        #self.topic_matrix1=tensor.ones(self.answer.shape[0] - 1,self.answer.shape[1]).flatten()-self.topic_matrix
        #self.topic_matrix_change=2*(self.topic_matrix-0.5)
        #self.topic_matrix_change=self.topic_matrix
        m = self.answer_mask[1:, :]
        #self.cost1 = -1.0 * tensor.dot(output_error_vector, m.flatten()*self.topic_matrix) / m.sum()
        self.cost = -1.0 * tensor.dot(output_error_vector, m.flatten()) / m.sum()
        #self.cost=(25*self.cost1+self.cost2)/26
        self.output_error_vector = output_error_vector.reshape([self.answer.shape[0] - 1 , self.answer.shape[1]]) 
        self.output_error_vector = self.output_error_vector * m
        self.output_error_vector = -output_error_vector.sum(axis=0) / m.sum(axis=0)
        
        self._set_parameters(input_params)  # params from list to TensorVirable
    

    def get_training_function(self, cr, error_type="RMSE", batch_size=10, batch_repeat=1):
        optimizer = self.options["optimizer"]
        lr = tensor.scalar(name='lr')
        grads = tensor.grad(self.cost, wrt=self.tparams.values())
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads,
                                            [self.question, self.question_mask,
                                             self.answer, self.answer_mask,self.style,self.style_mask, self.topic],
                                            [self.cost])
        
        def update_function(index):
            (question, question_mask), (answer, answer_mask),(style,style_mask),(topic,topic_mask), _, _ = \
                cr.get_train_set([index * batch_size, (index + 1) * batch_size])
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(question, question_mask, answer, answer_mask,style,style_mask,topic)
                f_update(self.options["learning_rate"])
            return cost
        
        return update_function
    

    def get_validing_function(self, cr):
        (question, question_mask), (answer, answer_mask),(style,style_mask),(topic,topic_mask), _, _ = cr.get_valid_set()
        #print topic
        valid_function = theano.function(inputs=[],
                                         outputs=[self.cost],
                                         givens={self.question: question,
                                                 self.question_mask: question_mask,
                                                 self.answer: answer,
                                                 self.answer_mask: answer_mask,
                                                 self.style:style,self.style_mask:style_mask,
                                                 self.topic :topic},
                                         name='valid_function',on_unused_input='ignore')
        
        return valid_function
    

    #def get_testing_function(self, cr):
        '''
        (question, question_mask), (answer, answer_mask),(style,style_mask),(topic,topic_mask), _, _ = cr.get_test_set()
        test_function = theano.function(inputs=[],
                                        outputs=[self.cost],
                                        givens={self.question: question,
                                                self.question_mask: question_mask,
                                                self.answer: answer,
                                                self.answer_mask: answer_mask,
                                                self.style:style,self.style_mask:style_mask,
                                                self.topic : topic},
                                        name='test_function',on_unused_input='ignore')
        '''
    def get_testing_function(self, cr,batch_size=10):
        test_function1 = theano.function(inputs=[self.question, self.question_mask,
                                             self.answer, self.answer_mask,self.style,self.style_mask, self.topic],
                                         outputs=[self.cost],
                                         name='test_function',on_unused_input='ignore')
        def test_function(index):
            #(question_all, question_mask_all), (answer_all, answer_mask_all),(style_all,style_mask_all),(topic_all,topic_mask_all), _, _ = cr.get_test_set([index * batch_size, (index + 1) * batch_size])
            (question, question_mask), (answer, answer_mask),(style,style_mask),(topic,topic_mask), _, _ = cr.get_test_set([index * batch_size, (index + 1) * batch_size])
            return test_function1(question, question_mask, answer, answer_mask,style,style_mask,topic)
        (question, question_mask), (answer, answer_mask),(style,style_mask),(topic,topic_mask), _, _ = cr.get_pr_set()
        pr_function = theano.function(inputs=[],
                                      outputs=[self.output_error_vector],
                                      givens={self.question: question,
                                              self.question_mask: question_mask,
                                              self.answer: answer,
                                              self.answer_mask: answer_mask,
                                              self.style:style,self.style_mask:style_mask,
                                              self.topic : topic},
                                      on_unused_input='ignore',
                                      name='pr_function')
        
        return test_function, pr_function
    

    def get_deploy_function(self):
        maxout_input1 = tensor.concatenate([self.decoder_hidden_status1[-1, :, :],
                                           self.answer_embedding[-1, :, :]],
                                          axis=1)
        #pred_word1, pred_word_probability1 = self.maxout_layer.getOutput(self.tparams['Wemb_e'], maxout_input)
        pred_word2, pred_word_probability2 = self.maxout_layer1.getOutput(self.tparams['Wemb_e'], maxout_input1)
        pred_word_probability=pred_word_probability2
        pred_words_array=theano.tensor.argsort(pred_word_probability)[:,-10:]
        pred_word_probability_array=theano.tensor.transpose(pred_word_probability[theano.tensor.arange(pred_words_array.shape[0]), theano.tensor.transpose(pred_words_array)])
        deploy_function = theano.function(inputs=[self.question, self.question_mask,
                                                  self.answer, self.answer_mask,self.style,self.style_mask,self.topic],
                                          outputs=[pred_words_array,pred_word_probability_array],
                                          on_unused_input='ignore',
                                          name='deploy_function')
        
        return deploy_function
    def get_cost(self):
        deploy_function = theano.function(inputs=[self.answer, self.answer_mask],
                                          outputs=self.cost)
        return deploy_function

    def get_encoder_vector(self):
        deploy_function = theano.function(inputs=[self.question, self.question_mask],
                                          outputs=self.encoder_hidden_status1[-1])
        return deploy_function
