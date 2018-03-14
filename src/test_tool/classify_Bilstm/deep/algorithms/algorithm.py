from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import heapq
import math

import theano
import theano.tensor as T

from deep.algorithms.util import numpy_floatX
from deep.dataloader.util import *



class algorithm:
    __metaclass__ = ABCMeta
    def __init__(self):
        self._params = None
    
    @abstractmethod
    def get_training_function(self):
        """
            :Return a theano function, which is a training fucntion whose
            input value is a index indicates the serial number of input mini-batch.
        """
        pass
    
    @abstractmethod
    def get_validing_function(self):
        """
            :Return a theano function which works on the valid data. The output of this fuction is similar 
            with @getTrainFunction, but without updating operation."""
        pass
    
    @abstractmethod
    def get_testing_function(self):
        """
            :Return a theano function which works on the test data. The output of this fuction is similar 
            with @getTrainFunction, but without updating operation."""
        pass

    @abstractmethod
    def get_deploy_function(self, param):
        """
            :Return a theano function, which is a testing function. Its 
            return value is (sentence embedding, predicting next sentence embedding, reference sentence embedding).
            In general, if the predicting next  embedding of sentence A is similar to the reference sentence 
            embedding of sentence B, we say that B is approximately next to A. """
        pass
    
    def _setParameters(self, params):
        if not params is None:
            for para0, para in zip(self.model_params, params):
                para0.set_value(para, borrow=True)
            
    def getParameters(self):
        return self.model_params
    
    @classmethod
    def transToTensor(cls, data, t):
        return theano.shared(
            numpy.array(
                data,
                dtype=t
            ),
            borrow=True
        )

    def init_tparams(self, params):
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams
    
    def adadelta(self, lr, tparams, grads, model_input, cost, givens=None):
        """
        An adaptive learning rate optimizer
    
        Parameters
        ----------
        lr : Theano SharedVariable
            Initial learning rate
        tpramas: Theano SharedVariable
            Model parameters
        grads: Theano variable
            Gradients of cost w.r.t to parameres
        input: Theano variable of input, list.
        cost: Theano variable
            Objective fucntion to minimize
    
        Notes
        -----
        For more information, see [ADADELTA]_.
    
        .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
           Rate Method*, arXiv:1212.5701.
        """
    
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.iteritems()]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                     name='%s_rup2' % k)
                       for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]
    
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
    
        f_grad_shared = theano.function(model_input, cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared', givens=givens)
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    
        f_update = theano.function([lr], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update',
                                   givens=givens)
    
        return f_grad_shared, f_update
        
    def rmsprop(self, lr, tparams, grads, model_input, cost, givens=None):
        """
        A variant of  SGD that scales the step size by running average of the
        recent step norms.
    
        Parameters
        ----------
        lr : Theano SharedVariable
            Initial learning rate
        tpramas: Theano SharedVariable
            Model parameters
        grads: Theano variable
            Gradients of cost w.r.t to parameres
        x: Theano variable
            Model inputs
        mask: Theano variable
            Sequence mask
        y: Theano variable
            Targets
        cost: Theano variable
            Objective fucntion to minimize
    
        Notes
        -----
        For more information, see [Hint2014]_.
    
        .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
           lecture 6a,
           http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """
    
        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.iteritems()]
        running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                       name='%s_rgrad' % k)
                         for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]
    
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
    
        f_grad_shared = theano.function(model_input, cost,
                                        updates=zgup + rgup + rg2up,
                                        name='rmsprop_f_grad_shared',
                                        givens=givens)
    
        updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                               name='%s_updir' % k)
                 for k, p in tparams.iteritems()]
        updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                     for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                                running_grads2)]
        param_up = [(p, p + udn[1])
                    for p, udn in zip(tparams.values(), updir_new)]
        f_update = theano.function([lr], [], updates=updir_new + param_up,
                                   on_unused_input='ignore',
                                   name='rmsprop_f_update', givens=givens)
    
        return f_grad_shared, f_update
    
    
    
def greed(sentence, cr, deploy_model):
    minLen = 5
    maxLen = 20
    (x, x_mask) = cr.transformInputData(sentence)
    print "original x: ", sentence
    currentLength = 0
    
    error = 0.0
    
    while True: 
        pred_w, pred_w_p = deploy_model(x, x_mask)
        pred_w_list = pred_w_p.flatten().tolist()
        
        sorted_index = heapq.nlargest(1, enumerate(pred_w_list), key=lambda s: s[1])
        p_word, p_word_prob = zip(*(sorted_index[:10]))
        # print 'currentLength %d:' % currentLength
        l1 = cr.transformInputText(p_word)
        l2 = p_word_prob
        # for ll1, ll2 in zip(l1, l2):
        #    print '%s\t%.8f' % (ll1, ll2)
        error -= numpy.log(l2)
        
        if currentLength < minLen:
            for c in p_word:
                choice = c
                if choice > 2:
                    break
        else:
            for c in p_word:
                choice = c
                if choice >= 2:
                    break
                
        x = numpy.concatenate([x, [[choice]]], axis=0)
        
        x_mask = numpy.concatenate([x_mask, [[1]]], axis=0)
        currentLength += 1
        if choice == 2 or currentLength > maxLen:
            break
#             print "pred: ", x
    return [x.flatten().tolist()[0]], error


def beam_search(sentence, cr, deploy_model):
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(12)
    
    maxLen = 12
    search_scope = 10
    beam_size = 18
    output_size = 5
    stop_tag = cr.dictionary['<END>']
    haveGone = set()
   
    class sentenceScorePair(object):
        def __init__(self, priority, sentence):
            self.priority = priority
            self.sentence = sentence
        def __cmp__(self, other):
            return cmp(self.priority, \
                       other.priority)
    
    def step(x0, score, pred_w_p):
        pred_w_list = pred_w_p.flatten().tolist()
        sorted_index = heapq.nlargest(search_scope, enumerate(pred_w_list), key=lambda s: s[1])
        result_list = list()
        candidates0 = list()
        
        for p_word, p_word_prob in sorted_index:
            new_x = x0 + [p_word]
            
            new_x_str = str(new_x)
            if new_x_str in haveGone:
                continue
            haveGone.add(new_x_str)
            
            if p_word == stop_tag:
                if len(x0) - base_length < minLen:
                    continue
                candidates0.append(sentenceScorePair(
                                score - math.log(p_word_prob),
                                new_x
                                ))
            else:
                result_list.append(sentenceScorePair(
                                score - math.log(p_word_prob),
                                new_x
                                ))
        return result_list, candidates0
    
    (x, x_mask) = cr.transformInputData(sentence)
    
    base_length = x.shape[0]
    if base_length > 3:
        minLen = 4
    else:
        minLen = 3
    
    pred_w, pred_w_p = deploy_model(x, x_mask)
    
    sentence = load_sentence(sentence, cr.dictionary, special_flag=cr.special_flag)
    sQueue, candidates = step(sentence, 0, pred_w_p)
    for iter in xrange(maxLen):
        current_len = len(sQueue)
        if current_len == 0:
            break
        
        buffer_Queue = list()
        
        candidate_list = [q.sentence for q in sQueue]
        x, x_mask = get_mask_data(candidate_list)
        pred_w, pred_w_p = deploy_model(x, x_mask)
        
        temp = pool.map(lambda i:step(sQueue[i].sentence, sQueue[i].priority, pred_w_p[i]), range(current_len))
        
        for b, c in temp:
            buffer_Queue.extend(b)
            candidates.extend(c)
        
        sQueue = buffer_Queue
        sQueue = heapq.nsmallest(beam_size, sQueue)
    
    candidates_buffer = list()
    for c in candidates:
        flag = False
        for i in xrange(1, len(c.sentence)):
            for j in xrange(i):
                if c.sentence[i] == c.sentence[j] and c.sentence[i] != stop_tag:
                    flag = True
                    break
            if flag:
                break
        if not flag:
            candidates_buffer.append(c)

    candidates = heapq.nsmallest(output_size, candidates_buffer)
    return [x.sentence for x in candidates], [x.priority for x in candidates]
