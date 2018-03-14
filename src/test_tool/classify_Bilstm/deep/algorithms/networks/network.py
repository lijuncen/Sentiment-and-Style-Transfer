import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy
import theano
import theano.tensor as T

from deep.dataloader.util import *
from deep.algorithms.util import numpy_floatX



class Network:
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

    
    def _set_parameters(self, param_dict):
        if param_dict is not None:
            for pname in self.tparams :
                assert pname in param_dict
                self.tparams[pname].set_value(param_dict[pname], borrow=True)
  
            
    def get_parameters(self):
        return self.tparams
  
    
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
    def sgd(self,lr, tparams, grads, input_data, cost,givens=None):
        """ Stochastic Gradient Descent
    
        :note: A more complicated version of sgd then needed.  This is
            done like that for adadelta and rmsprop.

        """
        # New set of shared variable that will contain the gradient
        # for a mini-batch.
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                for k, p in tparams.items()]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]

        # Function that computes gradients for a mini-batch, but do not
        # updates the weights.
        f_grad_shared = theano.function(input_data, cost, updates=gsup,
                                        name='sgd_f_grad_shared',
                                        givens=givens)

        pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

        # Function that updates the weights from the previously computed
        # gradient.
        f_update = theano.function([lr], [], updates=pup,
                                name='sgd_f_update',
                                givens=givens)

        return f_grad_shared, f_update