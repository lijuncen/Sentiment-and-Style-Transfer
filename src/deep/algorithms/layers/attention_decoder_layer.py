import numpy
import theano
import theano.tensor as tensor

from deep.algorithms.layers.layer import layer
from deep.algorithms.nodes.gru_node import GRUNode
from deep.algorithms.nodes.attention_node import AttentionNode
from deep.algorithms.util import numpy_floatX



class AttentionDecoderLayer(layer):

    def __init__(self, word_embedding_dim, hidden_status_dim, encoder_hidden_dim,
                 tparams, prefix='Decoder'):
        """
        Init the Decoder parameter: init_params.
        """
        self.hidden_status_dim = hidden_status_dim
        self.prefix = prefix
        self.node = GRUNode(word_embedding_dim=word_embedding_dim,
                            hidden_status_dim=hidden_status_dim,
                            tparams=tparams, prefix=self._p(self.prefix, 'GRU'))
        self.attention_node = AttentionNode(word_embedding_dim=word_embedding_dim,
                                            hidden_status_dim=hidden_status_dim,
                                            encoder_hidden_dim = encoder_hidden_dim,
                                            tparams=tparams, prefix=self._p(self.prefix, 'Attention'))


    def getOutput(self, inputs):
        """
        Get outputs of encoder layer.
        Return all of the hidden status.
        """
        (self.sentence, self.mask, self.encoder_hidden_status, self.question_mask) = inputs
        
        assert self.mask is not None
        n_steps = self.sentence.shape[0]
        if self.sentence.ndim == 3:
            n_samples = self.sentence.shape[1]
        else:
            n_samples = 1

        last_s = tensor.alloc(numpy_floatX(0.), n_samples, self.hidden_status_dim)
        state_below = self.sentence

        def upd(am_, x_, s_, h_, qm_):
            c = self.attention_node.node_update(s_, h_, qm_)
            x_ = tensor.dot(tensor.concatenate([x_, c], axis=1), self.node.get_params_W())
            s = self.node.node_update(am_, x_, s_)
            
            return s, c

        results, _ = theano.scan(upd,
                                 sequences=[self.mask, state_below],
                                 outputs_info=[last_s, None],
                                 non_sequences=[self.encoder_hidden_status, self.question_mask],
                                 name=self._p(self.prefix, '_scan'),
                                 n_steps=n_steps)
        hidden_status_outputs, context_outputs = results
        
        return hidden_status_outputs, context_outputs
