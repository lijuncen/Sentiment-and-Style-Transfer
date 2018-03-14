from theano import tensor as T, printing
import theano
import numpy as np
import config

__affinty_fun = None
__affinty_fun_gaussian = None

def __init():
    dataset = T.matrix("dataset", dtype=config.globalFloatType())
    trans_dataset = T.transpose(dataset)
    dot_mul = T.dot(dataset, trans_dataset)
    l2 = T.sqrt(T.sum(T.square(dataset), axis=1))
    
#     p =printing.Print("l2")
#     l2 = p(l2)
    
    l2_inv2 = T.inv(l2).dimshuffle(['x', 0])
#     p =printing.Print("l2_inv2")
#     l2_inv2 = p(l2_inv2)
    
    l2_inv1 = T.transpose(l2_inv2)
#     p =printing.Print("l2_inv1")
#     l2_inv1 = p(l2_inv1)
    
    l2_inv = T.dot(l2_inv1, l2_inv2)
    
#     p =printing.Print("l2_inv")
#     l2_inv = p(l2_inv)
    
    affinty = (T.mul(dot_mul, l2_inv) + 1) / 2
    globals()['__affinty_fun'] = theano.function(
             [dataset],
             [affinty],
             allow_input_downcast=True
             )
    
#     delta = 0.1
#     affinty_gaussian = T.exp(- (1-affinty) ** 2 / (2. * delta ** 2))
#     globals()['__affinty_fun_gaussian']  = theano.function(
#              [dataset],
#              [affinty_gaussian],
#              allow_input_downcast=True
#              )

def compute_affinity_matrix(dataset):
    dataset = np.asarray(dataset, dtype=config.globalFloatType())
    if globals()['__affinty_fun'] == None:
        __init()
    f = globals()['__affinty_fun']
    return f(dataset)


if __name__ == '__main__':
    testres = compute_affinity_matrix([ [1, 2, 3, 4, 5], \
                                                                 [2, 3, 4, 5, 5],
                                                                 [1, 4, 5, 6, 7],
                                                                 [1, 2, 3, 4, 5]])
#     testres = compute_affinity_gaussian_matrix([ [1, 2, 3, 4, 5], \
#                                                                  [2, 3, 4, 5, 5],
#                                                                  [1, 4, 5, 6, 7],
#                                                                  [1, 2, 3, 4, 5]])
    print testres
