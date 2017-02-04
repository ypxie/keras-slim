#directly taken from keras https://github.com/fchollet/keras

from __future__ import absolute_import
import .backend.export as K

def softmax(x):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim == 3:
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise Exception('Cannot apply softmax to a tensor that is not 2D or 3D. ' +
                        'Here, ndim=' + str(ndim))
                        
def softplus(x):
    return K.softplus(x)


def softsign(x):
    return K.softsign(x)


def relu(x, alpha=0., max_value=None):
    return K.relu(x, alpha=alpha, max_value=max_value)

def wrapper_elu(alpha=1):
    def f(x):
        pos = K.relu(x)
        neg = (x - K.abs(x)) * 0.5
        return pos + alpha * (K.exp(neg) - 1.)
    return f

def elu(x):
    return wrapper_elu(1)(x)
    
def tanh(x):
    return K.tanh(x)


def sigmoid(x):
    return K.sigmoid(x)


def hard_sigmoid(x):
    return K.hard_sigmoid(x)


def linear(x):
    '''
    The function returns the variable that is passed in, so all types work.
    '''
    return x

from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'activation function')
