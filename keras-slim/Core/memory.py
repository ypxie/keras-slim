
import numpy as np
import ..backend.export as T
from   ..backend.export import npwrapper
from   .frame import *
from ..utils.utils_func import slice_tensor
from ..utils import activations, initializations, regularizers



def read( w, M):
    return (w[:, :, None]*M).sum(axis=1)

def get_content_w( beta, k, M):
    num = beta[:, None] * cosine_distance(M, k)
    return T.softmax(num)

def get_location_w(g, s, C, gamma, wc, w_tm1):
    wg = g[:, None] * wc + (1-g[:, None])*w_tm1
    Cs = (C[None, :, :, :] * wg[:, None, None, :]).sum(axis=3)
    wtilda = (Cs * s[:, :, None]).sum(axis=1)
    #wout = renorm(wtilda ** gamma[:, None])
    #wout = T.softmax(wtilda ** gamma[:, None])
    print('we dont use gamma!')
    wout = renorm(wtilda)
    return wout

def get_controller_output(h, W_k, b_k, W_c, b_c, W_s, b_s, k_activ = T.tanh):
    k = k_activ(T.dot(h, W_k) + b_k)  # + 1e-6
    #k = theano.printing.Print('[Debug] k shape is: ', attrs=("shape",))(k)
    c = T.dot(h, W_c) + b_c
    beta = T.relu(c[:, 0],  max_value= 5) + 1e-4
    g = T.sigmoid(c[:, 1])
    gamma = T.relu(c[:, 2], max_value= 5) + 1.0001
    s = T.softmax(T.dot(h, W_s) + b_s)
    return k, beta, g, gamma, s

def wta(X):
    M = T.max(X, axis=-1, keepdims=True)
    R =T.switch(T.equal(X, M), X, 0.)
    return R

def renorm(x):
    return x / (x.sum(axis=1, keepdims=True)+0.000001)

def circulant(leng, n_shifts):
    """
    I confess, I'm actually proud of this hack. I hope you enjoy!
    This will generate a tensor with `n_shifts` of rotated versions the
    identity matrix. When this tensor is multiplied by a vector
    the result are `n_shifts` shifted versions of that vector. Since
    everything is done with inner products, everything is differentiable.
    Paramters:
    ----------
    leng: int > 0, number of memory locations, can be tensor variable
    n_shifts: int > 0, number of allowed shifts (if 1, no shift)
    Returns:
    --------
    shift operation, a tensor with dimensions (n_shifts, leng, leng)
    """
    #eye = np.eye(leng)
    #shifts = range(n_shifts//2, -n_shifts//2, -1)
    #C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    #return theano.shared(C.astype(theano.config.floatX))
    eye = T.eye(leng)
    shifts = range(n_shifts//2, -n_shifts//2, -1)
    C = T.stack([T.roll(eye, s, axis=1) for s in shifts], axis = 0)
    return C

def cosine_distance(M, k):
    #M = theano.printing.Print('[Debug] M shape is: ', attrs=("shape",))(M)
    dot = (M * k[:, None, :]).sum(axis=-1)
    nM = T.sqrt((M**2).sum(axis=-1))
    nk = T.sqrt((k**2).sum(axis=-1, keepdims=True))
    return dot / (nM * nk + 0.000001)

def quodra_distance(M,W, k):
    #M = theano.printing.Print('[Debug] M shape is: ', attrs=("shape",))(M)
    dot = (T.dot(M,W) * k[:, None, :]).sum(axis=-1)
    nM = T.sqrt((M**2).sum(axis=-1))
    nk = T.sqrt((k**2).sum(axis=-1, keepdims=True))
    return dot / (nM * nk + 0.0000001)