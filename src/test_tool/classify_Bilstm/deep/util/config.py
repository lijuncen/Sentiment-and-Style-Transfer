import theano


def globalFloatType():
    return theano.config.floatX

def globalCharSet() :
    return 'gb18030'