from theano.tensor import *
from .keras_backend.theano_backend import *
from .keras_backend.common import *

from theano import scan, shared, function
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
_FLOATX = 'float32'
floatX = _FLOATX
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

def assign_subtensor(dest, source, dest_slice):
    if hasattr(dest, '_keras_shape'):
        ks = dest._keras_shape
    dest = T.set_subtensor(dest[dest_slice], source)    
    if hasattr(dest, '_keras_shape'):
        dest._keras_shape = ks
    return dest    

def alloc(val, shape, broadcastable = True):
    output = T.alloc(val, *shape)
    if broadcastable == False:
        for axis in range(output.ndim):
            output = T.unbroadcast(output, axis)
    keras_shape = []
    for s in shape:
        if isinstance(s, int):
            keras_shape.append(s)
        else:
            keras_shape.append(None)
    output._keras_shape = tuple(keras_shape)
    return output
    
def isnan(x):
    return x==None

def reshape(x, shape):
    '''
    For this function, it is not possible to reliablly infer it's keras shape    
    '''
    output = T.reshape(x, shape)      
    return output


def normal(shape=None, mean=0.0, std=1.0, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = ()
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)

def uniform(shape=None, low=0.0, high=1.0, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = ()
    return rng.uniform(size = shape, low=low, high=high, dtype=dtype)

def binomial(shape=None, p=0.0, n=1,dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = ()
    return rng.binomial(size = shape, p=p, n= 1, dtype=dtype)

def multinomial(shape=None, pvals = 0.0, n =1, dtype=_FLOATX, rng=None):
    '''
    Sample n (n needs to be >= 1, default 1) times from a multinomial distribution 
    defined by probabilities pvals.

    Example : pvals = [[.98, .01, .01], [.01, .49, .50]] and n=1 will probably result 
    in [[1,0,0],[0,0,1]]. When setting n=2, this will probably result in [[2,0,0],[0,1,1]].

    Notes

    -size and ndim are only there keep the same signature as other uniform, binomial, normal, 
    etc. TODO : adapt multinomial to take that into account

    '''
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = ()
    return rng.multinomial(pvals=pvals,dtype=dtype)
   
   
