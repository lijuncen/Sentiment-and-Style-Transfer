import numpy
import theano
import theano.tensor as tensor

from deep.algorithms.nodes.node import Node
from deep.algorithms.util import ortho_weight, numpy_floatX, uniform_random_weight

from deep.util.config import globalFloatType


class AttentionNode(Node):

    def __init__(self, word_embedding_dim, hidden_status_dim, encoder_hidden_dim,
                 tparams=None, prefix='Attention'):
        """
        Init the GRU parameter: init_params.
        Updation in GRU :
            step1. r(t) = f(W_r dot x(t) + U_r dot h(t-1) + C_r dot h_last).
            step2. z(t) = f(W_z dot x(t) + U_z dot h(t-1) + C_z dot h_last).
            step3. h_wave(t) = f(W dot x(t) + U dot (r(t) * h(t-1)) + C dot h_last).
            step4. h(t) = (1-z(t)) * h(t-1) + z(t) * h_wave(t).
        We can combine W and C into one tensor W
        """
        self.hidden_status_dim = hidden_status_dim
        self.params = tparams
        self.prefix = prefix
        W_bound = numpy.sqrt(6. / (hidden_status_dim))

        # combine step1~3 W dot t, so W's dimension is (word_embedding_dim, hidden_status_dim, 3)
        W = uniform_random_weight(size=(hidden_status_dim, hidden_status_dim), bound=W_bound)
        # combine step1~2 U dot h, so U's dimension is (hidden_status_dim, 2)
        # connot combine step1~3, so split U_rh
        U = numpy.concatenate([ortho_weight(hidden_status_dim)]*int(encoder_hidden_dim/hidden_status_dim),
                              axis=0)
        # U = uniform_random_weight(height=2*hidden_status_dim, width=hidden_status_dim, bound=W_bound)
        va = numpy.zeros((hidden_status_dim,), dtype=globalFloatType())
        
        if tparams is not None: 
            tparams[self._p(prefix, 'W')] = theano.shared(W, name=self._p(prefix, 'W'))
            tparams[self._p(prefix, 'U')] = theano.shared(U, name=self._p(prefix, 'U'))
            tparams[self._p(prefix, 'va')] = theano.shared(va, name=self._p(prefix, 'va'))
        else:
            print ' tparams is None'
    

    def node_update(self, s_, h_, m_) :
        """
        Update params in Attention.
        """
        preact = tensor.dot(h_, self.params[self._p(self.prefix, 'U')])
        preact += tensor.addbroadcast(tensor.dot(s_, self.params[self._p(self.prefix, 'W')]).dimshuffle('x', 0, 1), 0)
        preact = tensor.dot(tensor.tanh(preact), self.params[self._p(self.prefix, 'va')]) * m_
        alpha = tensor.nnet.softmax(preact.dimshuffle(1, 0)).dimshuffle(1, 0, 'x')
#         pp = theano.printing.Print('alpha')
#         alpha = pp(alpha)
        c = (h_ * tensor.addbroadcast(alpha, 2)).sum(axis=0) # c is (samples,2*hidden)
    
        return c
