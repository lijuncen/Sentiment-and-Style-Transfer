# -*- encoding = utf-8 -*-
from collections import OrderedDict

import numpy
import theano
import theano.tensor as tensor
import theano.printing as printing

import deep.util.config as config
from deep.algorithms.networks.network import Network
from deep.algorithms.layers.rnn_encoder_layer import EncoderLayer
from deep.algorithms.layers.style_layer import StyleLayer
from deep.algorithms.layers.attention_decoder_layer import AttentionDecoderLayer

# from deep.algorithms.layers.softmax_layer import SoftmaxLayer
from deep.algorithms.layers.maxout_layer import MaxoutLayer

class BiEncoderAttentionDecoderStyleNetwork(Network):
    """
    This class will process the dialog pair with a encoder-decoder network.
    It has 2 abilities:
        1. Train the language model.
        2. Model the relationship of Q&A
        3. Model the attention
        4. Model the style
    """

    def init_global_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        params['Wemb_e'] = (0.01 * randn).astype(config.globalFloatType()) 
        randn = numpy.random.rand(options['n_words'], options['word_embedding_dim'])
        params['Wemb_d'] = (0.01 * randn).astype(config.globalFloatType()) 

        return params


    def __init__(self, n_words, word_embedding_dim, hidden_status_dim, style_number, style_dim, input_params=None):
        self.options = options = {
            'n_words': n_words,
            'word_embedding_dim': word_embedding_dim,
            'hidden_status_dim': hidden_status_dim,
            'style_number': style_number,
            'style_dim': style_dim,
            'learning_rate': 0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
            'optimizer': self.rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
            }
        # global paramters.
        params = self.init_global_params(options)
        # Theano paramters,
        self.tparams = self.init_tparams(params)
        # Used for dropout.
        # self.use_noise = theano.shared(numpy_floatX(0.))

        # construct network
        theano.config.compute_test_value = 'off'
        self.question = tensor.matrix('question', dtype='int64')
        self.question_mask = tensor.matrix('question_mask', dtype=config.globalFloatType())
        # self.question.tag.test_value = numpy.array([[10, 2, 0], [5, 9, 2]]) # for debug
        # self.question_mask.tag.test_value = numpy.array([[1, 1, 0], [1, 1, 1]]) # for debug
        self.question_embedding = self.tparams['Wemb_e'][self.question.flatten()].reshape(
            [self.question.shape[0], self.question.shape[1], options['word_embedding_dim']])
        #   1. forward encoder layer
        self.forward_encoder_layer = EncoderLayer(word_embedding_dim=options['word_embedding_dim'],
                                                  hidden_status_dim=options['hidden_status_dim'],
                                                  tparams=self.tparams, prefix='forward_Encoder')
        self.forward_encoder_hidden_status = \
            self.forward_encoder_layer.getOutput(inputs=(self.question_embedding, self.question_mask))
            
        #   2. backward encoder layer
        self.backward_encoder_layer = EncoderLayer(word_embedding_dim=options['word_embedding_dim'],
                                                   hidden_status_dim=options['hidden_status_dim'],
                                                   tparams=self.tparams, prefix='backward_Encoder')
        self.backward_encoder_hidden_status = \
            self.backward_encoder_layer.getOutput(inputs=(self.question_embedding[::-1, :, :],
                                                          self.question_mask[::-1, :]))
        self.encoder_hidden_status = tensor.concatenate([self.forward_encoder_hidden_status,
                                                         self.backward_encoder_hidden_status[::-1, :, :]],
                                                        axis=2)
        
        #   3. decoder layer
        self.answer = tensor.matrix('answer', dtype='int64')
        self.answer_mask = tensor.matrix('answer_mask', dtype=config.globalFloatType())
        # self.answer.tag.test_value = numpy.array([[11, 10, 2], [5, 2, 0]]) # for debug
        # self.answer_mask.tag.test_value = numpy.array([[1, 1, 1], [1, 1, 0]]) # for debug
        self.answer_embedding = self.tparams['Wemb_d'][self.answer.flatten()].reshape(
            [self.answer.shape[0], self.answer.shape[1], options['word_embedding_dim']])
        self.decoder_layer = \
            AttentionDecoderLayer(word_embedding_dim=options['word_embedding_dim'] + 2 * options['hidden_status_dim'],
                                  hidden_status_dim=options['hidden_status_dim'],
                                  encoder_hidden_dim=2 * options['hidden_status_dim'],
                                  tparams=self.tparams, prefix='Decoder')
        self.decoder_hidden_status, self.context = \
            self.decoder_layer.getOutput(inputs=(self.answer_embedding, self.answer_mask,
                                                 self.encoder_hidden_status, self.question_mask))
        #   4. style  layer
        
        self.style_layer = StyleLayer(style_number, style_dim, 3 * hidden_status_dim, self.tparams, 'style')
        
        question_style_input_embedding1 = self.forward_encoder_hidden_status[-1, :, :].reshape([self.answer.shape[1], options['hidden_status_dim']])
        question_style_input_embedding2 = self.backward_encoder_hidden_status[-1, :, :].reshape([self.answer.shape[1], options['hidden_status_dim']])
        answer_style_input_embedding = self.backward_encoder_hidden_status[-1, :, :].reshape([self.answer.shape[1], options['hidden_status_dim']])
        
        pair_embedding = tensor.concatenate([question_style_input_embedding1,
                                                                             question_style_input_embedding2,
                                                                             answer_style_input_embedding],
                                                                             axis=1)
        self.style = self.style_layer.getOutput(pair_embedding)
        
        
        #   5. maxout  layer
        self.maxout_layer = MaxoutLayer(base_dim=options['word_embedding_dim'],
                                                    refer_dim=3 * options["hidden_status_dim"] + options['word_embedding_dim'] + options['style_dim'],
                                                    tparams=self.tparams,
                                                    prefix="maxout")
        
        maxout_decoder_hidden_status_input = self.decoder_hidden_status[:-1, :, :].\
                                        reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                         options['hidden_status_dim']])
        
        maxout_context_hidden_status_input = self.context[:-1, :, :].\
                                reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                         2 * options['hidden_status_dim']])
        
        maxout_answer_wordEmbedding_input = self.answer_embedding[:-1, :, :].\
                                reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                         options['word_embedding_dim']])
        
        maxout_answer_style_input = tensor.alloc(self.style.dimshuffle('x', 0, 1),
                                                                                    self.answer.shape[0] - 1,
                                                                                    self.answer.shape[1],
                                                                                    options['style_dim'])
        
        maxout_answer_style_input = maxout_answer_style_input.\
                                                                                    reshape([(self.answer.shape[0] - 1) * self.answer.shape[1],
                                                                                     options['style_dim']])
                                                                                    
        self.maxout_input = \
            tensor.concatenate(\
                               [maxout_decoder_hidden_status_input,
                                maxout_context_hidden_status_input,
                                maxout_answer_wordEmbedding_input,
                                maxout_answer_style_input],
                                axis=1)
        output_error_vector = self.maxout_layer.negative_log_likelihood(
                                                                     self.tparams['Wemb_d'],
                                                                     self.maxout_input,
                                                                     y=self.answer[1:, :].flatten())
        m = self.answer_mask[1:, :]
        self.cost = -1.0 * tensor.dot(output_error_vector, m.flatten()) / m.sum()
        self.output_error_vector = output_error_vector.reshape([self.answer.shape[0] - 1 , self.answer.shape[1]]) 
        self.output_error_vector = self.output_error_vector * m
        self.output_error_vector = -self.output_error_vector.sum(axis=0) / m.sum(axis=0)
        
        self._set_parameters(input_params)  # params from list to TensorVirable
    

    def get_training_function(self, cr, error_type="RMSE", batch_size=10, batch_repeat=1):
        optimizer = self.options["optimizer"]
        lr = tensor.scalar(name='lr')
        grads = tensor.grad(self.cost, wrt=self.tparams.values())
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads,
                                            [self.question, self.question_mask,
                                             self.answer, self.answer_mask],
                                            [self.cost])
        
        def update_function(index):
            (question, question_mask), (answer, answer_mask), _, _ = \
                cr.get_train_set([index * batch_size, (index + 1) * batch_size])
            for _ in xrange(batch_repeat):
                cost = f_grad_shared(question, question_mask, answer, answer_mask)
                f_update(self.options["learning_rate"])
            return cost
        
        return update_function
    

    def get_validing_function(self, cr):
        (question, question_mask), (answer, answer_mask), _, _ = cr.get_valid_set()
        valid_function = theano.function(inputs=[],
                                         outputs=[self.cost],
                                         givens={self.question: question,
                                                 self.question_mask: question_mask,
                                                 self.answer: answer,
                                                 self.answer_mask: answer_mask},
                                         name='valid_function')
        
        return valid_function
    

    def get_testing_function(self, cr):
        (question, question_mask), (answer, answer_mask), _, _ = cr.get_test_set()
        test_function = theano.function(inputs=[],
                                        outputs=[self.cost],
                                        givens={self.question: question,
                                                self.question_mask: question_mask,
                                                self.answer: answer,
                                                self.answer_mask: answer_mask},
                                        name='test_function')
        (question, question_mask), (answer, answer_mask), _, _ = cr.get_pr_set()
        pr_function = theano.function(inputs=[],
                                      outputs=[self.output_error_vector],
                                      givens={self.question: question,
                                              self.question_mask: question_mask,
                                              self.answer: answer,
                                              self.answer_mask: answer_mask},
                                      name='pr_function')
        
        return test_function, pr_function
    

    def get_deploy_function(self, style_type):
        maxout_input = tensor.concatenate([self.decoder_hidden_status[-1, :, :],
                                            self.encoder_hidden_status[-1, :, :],
                                            self.answer_embedding[-1, :, :]],
                                           axis=1)
        b = self.style_layer.getStyleMatrix().get_value()[style_type]
        
        pred_word, pred_word_probability = self.maxout_layer.getOutput(self.tparams['Wemb_d'], maxout_input)
        deploy_function = theano.function(inputs=[self.question, self.question_mask,
                                                  self.answer, self.answer_mask],
                                          outputs=[pred_word, pred_word_probability],
                                          givens={self.style: b},
                                          name='deploy_function')
        
        return deploy_function
