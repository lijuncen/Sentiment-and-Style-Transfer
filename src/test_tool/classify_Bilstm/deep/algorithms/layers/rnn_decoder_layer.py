import theano
import theano.tensor as tensor

from deep.algorithms.layers.layer import layer
from deep.algorithms.nodes.gru_node import GRUNode
from deep.algorithms.util import numpy_floatX

    

class DecoderLayer_Seq2Seq(layer):

    def __init__(self, word_embedding_dim=128, hidden_status_dim=128, tparams=None):
        """
        Init the Decoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.prefix = 'Decoder'
        self.node = GRUNode(word_embedding_dim=word_embedding_dim, 
                            hidden_status_dim=hidden_status_dim, 
                            tparams=tparams, prefix=self.prefix)

    
    def getOutput(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask, self.forward_hidden_status) = inputs

        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        last_h = tensor.alloc(self.forward_hidden_status[-1,:,:], n_samples, self.hidden_status_dim)
        state_below = tensor.dot(self.sentence, self.node.get_params_W())

        results, _ = theano.scan(self.node.node_update,
                                 sequences=[self.mask, state_below],
                                 outputs_info=[last_h],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs = results
        
        return hidden_status_outputs

    

class DecoderLayer_Cho(layer):

    def __init__(self, word_embedding_dim=128, hidden_status_dim=128, tparams=None):
        """
        Init the Decoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.prefix = 'Decoder'
        self.node = GRUNode(word_embedding_dim=word_embedding_dim, 
                            hidden_status_dim=hidden_status_dim, 
                            tparams=tparams, prefix=self.prefix)

    
    def getOutput(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask, self.forward_hidden_status) = inputs

        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        last_h = tensor.alloc(numpy_floatX(0.), n_samples, self.hidden_status_dim)
        state_below = tensor.dot(tensor.concatenate([self.sentence, 
                                                     tensor.alloc(self.forward_hidden_status[-1,:,:],
                                                                  self.sentence.shape[0], 
                                                                  self.forward_hidden_status.shape[1],
                                                                  self.forward_hidden_status.shape[2])],
                                                    axis=2),
                                 self.node.get_params_W())

        results, _ = theano.scan(self.node.node_update,
                                 sequences=[self.mask, state_below],
                                 outputs_info=[last_h],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs = results
        
        return hidden_status_outputs