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
from algorithms.layers.softmax_layer import SoftmaxLayer
from deep.algorithms.layers.maxout_layer import MaxoutLayer
from deep.algorithms.layers.rnn_encoder_layer import EncoderLayer
from deep.algorithms.layers.rnn_decoder_layer import DecoderLayer_Cho
import string
import theano.tensor as T


class RnnEncoderDecoderNetwork(Network):
    """
    This class will process the dialog pair with a encoder-decoder network.
    It has 2 abilities:
        1. Train the language model.
        2. Model the relationship of Q&A
    """

    def init_global_params(self, options,word_embedings):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        params['Wemb_e'] = (0.1 * randn).astype(config.globalFloatType()) 
        #params['Wemb_e'] =word_embedings
        #randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        #params['Wemb_e'] = (0.1 * randn).astype(config.globalFloatType())
        randn = numpy.random.rand(options['hidden_status_dim'], options['hidden_status_dim'],options['hidden_status_dim'])
        params['P_M']= (0.1 * randn).astype(config.globalFloatType())
        randn = numpy.random.rand(2*options['hidden_status_dim'], options['hidden_status_dim'])
        params['P_N']= (0.1 * randn).astype(config.globalFloatType())
        '''
        randn = numpy.random.rand(1)
        params['P_alpha']= (1 * randn).astype(config.globalFloatType())
        randn = numpy.random.rand(1)
        params['P_beta']= (1 * randn).astype(config.globalFloatType())
        '''
        #randn = numpy.random.rand(options['topic_embedding_dim'], options['topic_embedding_dim'])/options['topic_embedding_dim']*2
        #params['QTA']=(1.0 * randn).astype(config.globalFloatType())
        #randn = numpy.random.rand(options['n_topics'], options['topic_embedding_dim'])
        #params['Temb'] = (0.1 * randn).astype(config.globalFloatType())
        #params['Temb'] = numpy.dot(params['Qemb'],params['QTA'])
        return params


    def __init__(self, n_words, word_embedding_dim=128, hidden_status_dim=128, n_topics=2, topic_embedding_dim=5,input_params=None,word_embedings=None):
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
        params = self.init_global_params(options,word_embedings)
        # Theano paramters,
        self.tparams = self.init_tparams(params)
        #print self.tparams['Temb'] 
        #self.answer_emb=T.dot(self.tparams['Qemb'],self.tparams['QTA'])
        # Used for dropout.
        # self.use_noise = theano.shared(numpy_floatX(0.))

        # construct network
        theano.config.compute_test_value = 'off'
        self.reference = tensor.matrix('reference', dtype='int64')
        self.reference_mask = tensor.matrix('reference_mask', dtype=config.globalFloatType())
        #self.reference_mask = tensor.matrix('reference_mask', dtype='int64')
        self.topic = tensor.matrix('topic', dtype=config.globalFloatType())
        self.context=tensor.tensor3('context', dtype='int64')
        self.context_mask = tensor.tensor3('context_mask', dtype=config.globalFloatType())
        self.context_mask2 = tensor.matrix('context_mask2', dtype=config.globalFloatType())
        # self.reference.tag.test_value = numpy.array([[10, 2, 0], [5, 9, 2]]) # for debug
        # self.reference_mask.tag.test_value = numpy.array([[1, 1, 0], [1, 1, 1]]) # for debug
        self.reference_embedding = self.tparams['Wemb_e'][self.reference.flatten()].reshape(
            [self.reference.shape[0], self.reference.shape[1], options['word_embedding_dim']])
        #   1. encoder layer
        self.encoder_layer_reference = EncoderLayer(word_embedding_dim=options['word_embedding_dim'],
                                          hidden_status_dim=options['hidden_status_dim'],
                                          tparams=self.tparams,prefix='Encoder')
        self.encoder_hidden_status_reference = self.encoder_layer_reference.getOutput(inputs=(self.reference_embedding, self.reference_mask))
        #self.topic_states = self.tparams['Temb'][self.topic.flatten()].reshape([1,self.reference.shape[1], options['topic_embedding_dim']])
        #self.topic_change=T.alloc(self.topic_states,self.reference.shape[0], self.reference.shape[1], options['topic_embedding_dim'])
        #self.encoder_hidden_status = T.concatenate([self.encoder_hidden_status,self.topic_change], axis=2)
        #   2. decoder layer
        self.answer = tensor.matrix('answer', dtype='int64')
        self.answer_mask = tensor.matrix('answer_mask', dtype=config.globalFloatType())
        # self.answer.tag.test_value = numpy.array([[11, 10, 2], [5, 2, 0]]) # for debug
        # self.answer_mask.tag.test_value = numpy.array([[1, 1, 1], [1, 1, 0]]) # for debug
        self.answer_embedding = self.tparams['Wemb_e'][self.answer.flatten()].reshape(
            [self.answer.shape[0], self.answer.shape[1], options['word_embedding_dim']])
        self.encoder_hidden_status_answer = self.encoder_layer_reference.getOutput(inputs=(self.answer_embedding, self.answer_mask))
        self.context_emdedding= self.tparams['Wemb_e'][self.context.flatten()].reshape(
            [self.context.shape[0], self.context.shape[1]*self.context.shape[2], options['word_embedding_dim']])
        self.encoder_hidden_status_context1 = self.encoder_layer_reference.getOutput(inputs=(self.context_emdedding, self.context_mask.flatten().reshape([self.context.shape[0], self.context.shape[1]*self.context.shape[2]])))
        self.encoder_layer_context2 = EncoderLayer(word_embedding_dim=options['hidden_status_dim'],
                                          hidden_status_dim=options['hidden_status_dim'],
                                          tparams=self.tparams,prefix='Encoder2')
        self.encoder_hidden_status_context2=self.encoder_layer_context2.getOutput(inputs=(self.encoder_hidden_status_context1[-1,:,:].reshape([self.context.shape[1], self.context.shape[2],options['hidden_status_dim']]), self.context_mask2))
        #self.context_processed=tensor.transpose(tensor.transpose(self.encoder_hidden_status_context2[-1])*self.topic.flatten())
        self.context_processed=self.encoder_hidden_status_context2[-1]
        #self.rcm=tensor.dot(tensor.concatenate([self.encoder_hidden_status_reference[-1],self.context_processed],axis=1),self.tparams['P_M'])
        #self.acm=tensor.dot(tensor.concatenate([self.encoder_hidden_status_answer[-1],self.context_processed],axis=1),self.tparams['P_M'])
        #self.softmax_input=tensor.dot(tensor.concatenate([self.acm,self.rcm],axis=1),self.tparams['P_N'])
        self.rmc=tensor.batched_dot(tensor.dot(self.encoder_hidden_status_reference[-1],self.tparams['P_M']),self.context_processed)
        self.amc=tensor.batched_dot(tensor.dot(self.encoder_hidden_status_answer[-1],self.tparams['P_M']),self.context_processed)
        self.softmax_input=tensor.dot(tensor.concatenate([self.rmc,self.amc],axis=1),self.tparams['P_N'])
        #self.softmax_input=self.encoder_hidden_status_reference[-1]-self.encoder_hidden_status_reference[-1]+self.encoder_hidden_status_context2[-1]-self.encoder_hidden_status_answer[-1]+self.encoder_hidden_status_answer[-1]
        #self.softmax_input=self.rcm-self.acm
        #self.softmax_layer=SoftmaxLayer(n_in=options['hidden_status_dim'],n_out=3,tparams=self.tparams)
        self.softmax_layer=SoftmaxLayer(n_in=options['hidden_status_dim'],n_out=3,tparams=self.tparams)
	self.output_vector=self.softmax_layer.negative_log_likelihood(self.softmax_input,tensor.cast(self.topic.flatten()+tensor.ones_like(self.topic.flatten()),'int64'))
        self.cost=-1.0*self.output_vector.sum()/self.context.shape[2]
        #self.cost=((tensor.dot(mutti_m_am,(score-topic.flatten()))**2).sum()+0.01*self.l2)/(self.context.shape[2]/2)
        #self.cost=((tensor.max(tensor.dot(mutti_m_am,(topic.flatten()-score))*tensor.sgn(tensor.dot(mutti_m_am,(topic.flatten()))-tensor.ones(self.context.shape[2]/2)/2),tensor.zeros(self.context.shape[2]/2))**2).sum()+0.01*self.l2)/(self.context.shape[2]/2)
        '''
        self.ground_truth=tensor.dot(mutti_m_am,topic.flatten())
        self.score_diff=tensor.dot(mutti_m_am,score)
        self.ground_minus_score=(self.ground_truth-self.score_diff)
        #self.cost_max=(tensor.max(tensor.zeros_like(self.ground_truth),self.ground_truth*self.ground_minus_score))**2
        self.cost_max=(tensor.max(tensor.concatenate(([tensor.zeros_like(self.ground_truth)],[self.ground_truth*self.ground_minus_score]),axis=0),axis=0))**2+(tensor.ones_like(self.ground_truth)-tensor.abs_(self.ground_truth))*(self.ground_minus_score)**2
        self.cost=(self.cost_max.sum()+0.01*self.l2)/(self.context.shape[2]/2)
        '''
	#self.cost=((tensor.dot(mutti_m_am,(score-topic.flatten()))**2).sum()+((score-topic.flatten())**2).sum()+0.01*self.l2)/(self.context.shape[2]/2)
        '''
        self.decoder_layer = DecoderLayer_Cho(word_embedding_dim=options['word_embedding_dim'] + options['hidden_status_dim'],
                                              hidden_status_dim=options['hidden_status_dim'],
                                              tparams=self.tparams)
        self.decoder_hidden_status = self.decoder_layer.getOutput(inputs=(self.answer_embedding, self.answer_mask,
                                                                          self.encoder_hidden_status))

        #   3. maxout  layer
        self.maxout_layer = MaxoutLayer(base_dim=options['word_embedding_dim'],
                                                    refer_dim=2 * options["hidden_status_dim"] + options['word_embedding_dim'],
                                                    tparams=self.tparams,
                                                    prefix="maxout")
        self.maxout_input = tensor.concatenate([self.decoder_hidden_status[:-1, :, :].
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             options['hidden_status_dim']]),
                                                 tensor.alloc(self.encoder_hidden_status[-1, :, :],
                                                              self.answer.shape[0] - 1,
                                                              self.answer.shape[1],
                                                              options['hidden_status_dim']).
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             options['hidden_status_dim']]),
                                                 self.answer_embedding[:-1, :, :].
                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                             options['word_embedding_dim']])],
                                                axis=1)
        output_error_vector = self.maxout_layer.negative_log_likelihood(self.tparams['Wemb_d'],
                                                                    self.maxout_input,
                                                                    y=self.answer[1:, :].flatten())
        self.topic_matrix=tensor.alloc(self.topic.flatten(),self.answer.shape[0] - 1,self.answer.shape[1]).flatten()
        #self.topic_matrix_change=2*(self.topic_matrix-0.5)
        self.topic_matrix_change=self.topic_matrix
        m = self.answer_mask[1:, :]
        self.cost = -1.0 * tensor.dot(output_error_vector, m.flatten()*self.topic_matrix_change) / m.sum()
        self.output_error_vector = output_error_vector.reshape([self.answer.shape[0] - 1 , self.answer.shape[1]])
        self.output_error_vector = self.output_error_vector * m
        self.output_error_vector = -output_error_vector.sum(axis=0) / m.sum(axis=0)
        '''
        self.output_error_vector=self.cost
        self._set_parameters(input_params)  # params from list to TensorVirable


    def get_training_function(self, cr, error_type="RMSE", batch_size=10, batch_repeat=1):
        optimizer = self.options["optimizer"]
        lr = tensor.scalar(name='lr')
        grads = tensor.grad(self.cost, wrt=self.tparams.values())
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads,
                                            [self.reference, self.reference_mask,
                                             self.answer, self.answer_mask, self.topic,self.context,self.context_mask,self.context_mask2],
                                            [self.cost])

        def update_function(index):
            (reference, reference_mask), (answer, answer_mask),(topic,topic_mask),(context,context_mask,context_mask2), _, _ = \
                cr.get_train_set([index * batch_size, (index + 1) * batch_size])
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(reference, reference_mask, answer, answer_mask,topic,context,context_mask,context_mask2)
                f_update(self.options["learning_rate"])
            return cost

        return update_function


    def get_validing_function(self, cr):
        (reference, reference_mask), (answer, answer_mask),(topic,topic_mask),(context,context_mask,context_mask2), _, _ = cr.get_valid_set()
        #print len(reference[0])
        #print len(answer[0])
        #print len(topic[0])
        #print len(context[0])
        #print topic
        valid_function = theano.function(inputs=[],
                                         outputs=[self.cost],
                                         givens={self.reference: reference,
                                                 self.reference_mask: reference_mask,
                                                 self.answer: answer,
                                                 self.answer_mask: answer_mask,
                                                 self.topic :topic,
                                                 self.context :context,
                                                 self.context_mask:context_mask,
                                                 self.context_mask2:context_mask2},
                                         name='valid_function')

        return valid_function


    def get_testing_function(self, cr):
        (reference, reference_mask), (answer, answer_mask),(topic,topic_mask),(context,context_mask,context_mask2), _, _ = cr.get_test_set()
        test_function = theano.function(inputs=[],
                                        outputs=[self.cost],
                                        givens={self.reference: reference,
                                                self.reference_mask: reference_mask,
                                                self.answer: answer,
                                                self.answer_mask: answer_mask,
                                                self.topic : topic,
                                                self.context :context,
                                                self.context_mask:context_mask,
                                                self.context_mask2:context_mask2},
                                        name='test_function')
        (reference, reference_mask), (answer, answer_mask),(topic,topic_mask),(context,context_mask,context_mask2), _, _ = cr.get_pr_set()
        #print context,context_mask,context_mask2
        pr_function = theano.function(inputs=[],
                                      outputs=[self.output_error_vector],
                                      givens={self.reference: reference,
                                              self.reference_mask: reference_mask,
                                              self.answer: answer,
                                              self.answer_mask: answer_mask,
                                              self.topic : topic,
                                              self.context :context,
                                              self.context_mask:context_mask,
                                              self.context_mask2:context_mask2},
                                      on_unused_input='ignore',
                                      name='pr_function')

        return test_function, pr_function


    def get_deploy_function(self):
        maxout_input = tensor.concatenate([self.decoder_hidden_status[-1, :, :],
                                           self.encoder_hidden_status[-1, :, :],
                                           self.answer_embedding[-1, :, :]],
                                          axis=1)
        pred_word, pred_word_probability = self.maxout_layer.getOutput(self.tparams['Wemb_d'], maxout_input)
        pred_words_array=theano.tensor.argsort(pred_word_probability)[:,-10:]
        pred_word_probability_array=theano.tensor.transpose(pred_word_probability[theano.tensor.arange(pred_words_array.shape[0]), theano.tensor.transpose(pred_words_array)])
        deploy_function = theano.function(inputs=[self.reference, self.reference_mask,
                                                  self.answer, self.answer_mask,self.topic],
                                          outputs=[pred_words_array,pred_word_probability_array],
                                          on_unused_input='ignore',
                                          name='deploy_function')

        return deploy_function

    def classification_deploy(self):
        pred_word, pred_word_probability = self.softmax_layer.getOutput(self.softmax_input)
        deploy_function = theano.function(inputs=[self.reference, self.reference_mask,
                                                  self.answer, self.answer_mask,self.context,self.context_mask,self.context_mask2],
                                          outputs=[pred_word],
                                          on_unused_input='ignore',
                                          name='deploy_function')

        return deploy_function



    def get_cost(self):
        deploy_function = theano.function(inputs=[self.reference, self.reference_mask,
                                                  self.answer, self.answer_mask,self.context,self.context_mask,self.context_mask2],
                                          outputs=self.score)
        return deploy_function
