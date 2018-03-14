import numpy
import theano
import theano.tensor as T

from deep.util.config import globalFloatType
import deep.algorithms.util as util
from deep.algorithms.util import ortho_weight, numpy_floatX, uniform_random_weight
from layer import layer

import theano.printing as printing

class StyleLayer(layer):
    
    def __init__(self, style_number, style_dim, hidden_status_dim, tparams=None, prefix='style'):
        """
        Init the style parameter: init_params.
        """
        self.W_BBeta = theano.shared(
            value=util.uniform_random_weight((style_dim, style_dim), 0.1),
            name=self._p(prefix, 'W_BBeta'),
            borrow=True
        )
        self.W_hBeta = theano.shared(
            value=util.uniform_random_weight((hidden_status_dim, style_dim), 0.1),
            name=self._p(prefix, 'W_hBeta'),
            borrow=True
        )
        self.B = theano.shared(
            value=util.uniform_random_weight((style_number, style_dim), 0.1),
            name=self._p(prefix, 'B'),
            borrow=True
        )
        self.vb = theano.shared(
                                util.uniform_random_weight((style_dim,), 0.1),
                                name=self._p(prefix, 'vb'),
                                borrow=True
                                )
        
        # parameters of the model
        self.params = [self.W_BBeta, self.W_hBeta, self.B, self.vb]
        if not tparams is None:
            tparams[self._p(prefix, 'W_BBeta')] = self.W_BBeta
            tparams[self._p(prefix, 'W_hBeta')] = self.W_hBeta
            tparams[self._p(prefix, 'B')] = self.B
            tparams[self._p(prefix, 'vb')] = self.vb
        else:
            print " tparams is None"

    
    def getOutput(self, inputs):
        """
        Decide the weights of each status over the style matrix.
        Return the style embedding for each status
        """
        h_wave = T.dot(inputs, self.W_hBeta)
        h_wave = h_wave.dimshuffle(0, 'x', 1)
        
        B_wave = T.dot(self.B, self.W_BBeta)
        B_wave = B_wave.dimshuffle('x', 0, 1)
        
        beta = h_wave + B_wave
        
        beta = T.tanh(beta)
        beta = T.dot(beta, self.vb)
        beta = T.nnet.softmax(beta)
        
        b = T.dot(beta, self.B)

#         beta_pred = T.argmax(beta, axis=1)
#         b = self.B[beta_pred]
        
        return b
    
    
    def getStyleMatrix(self):
        return self.B
