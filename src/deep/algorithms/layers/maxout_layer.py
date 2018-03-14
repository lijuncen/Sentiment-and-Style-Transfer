import theano
import theano.tensor as T
#import theano.tensor.signal.downsample as downsample
import theano.tensor.signal.pool as pool

from layer import layer
import deep.algorithms.util as util


class MaxoutLayer(layer):
    '''
    This class maxout each row of a matrix.
    '''
    def __init__(self, base_dim, refer_dim, tparams, prefix="maxout"):
        self.W_t = theano.shared(
            value=util.uniform_random_weight((refer_dim, 2 * refer_dim), 0.1),
            name=self._p(prefix, 'W_t'),
            borrow=True
        )
        self.W_o = theano.shared(
            value=util.uniform_random_weight((base_dim, refer_dim), 0.1),
            name=self._p(prefix, 'W_o'),
            borrow=True
        )
        # parameters of the model
        self.params = [self.W_t, self.W_o]
        if not tparams is None:
            tparams[self._p(prefix, 'W_t')] = self.W_t
            tparams[self._p(prefix, 'W_o')] = self.W_o
        else:
            print " tparams is None"


    def getOutput(self, base_data, refer_data):
        t_wave = T.dot(refer_data, self.W_t)
        #t = downsample.max_pool_2d(t_wave, ds=(1, 2), ignore_border=True)
        t = pool.pool_2d(t_wave, ds=(1, 2), mode='max', ignore_border=True)
        
        # o = T.dot(base_data, self.W_o)
        # p_y_given_x = T.dot(t, T.transpose(o))
        p_y_given_x = T.dot(T.dot(t, T.transpose(self.W_o)), T.transpose(base_data))
        p_y_given_x = T.nnet.softmax(p_y_given_x*2)
        y_pred = T.argmax(p_y_given_x, axis=1)  # useless in deploy, but commenting it won't get faster performance
        return y_pred, p_y_given_x


    def negative_log_likelihood(self, base_data, refer_data, y):
        t_wave = T.dot(refer_data, self.W_t)
        #t = downsample.max_pool_2d(t_wave, ds=(1, 2), ignore_border=True)
        t = pool.pool_2d(t_wave, ds=(1, 2), mode='max', ignore_border=True)

        # o = T.dot(base_data, self.W_o)        
        # p_y_given_x = T.dot(t, T.transpose(o))

        p_y_given_x = T.dot(T.dot(t, T.transpose(self.W_o)), T.transpose(base_data))
        p_y_given_x = T.nnet.softmax(p_y_given_x)
#         y_pred = T.argmax(p_y_given_x, axis=1)
        error_vector = T.log(p_y_given_x)[T.arange(y.shape[0]), y]
        return error_vector
