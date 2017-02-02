#directly taken from keras https://github.com/fchollet/keras
from __future__ import absolute_import
import numpy as np
import .backend.export as K

def get_fans(shape, dim_ordering='th',**kwargs):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def ortho_weight(shape, **kwargs):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    nin,nout = shape[0], shape[1]
    W = np.random.randn(nin,nout)
    
    u, _, v = np.linalg.svd(W, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.reshape(shape)

    return q.astype(K.floatX)

def norm_weight(shape, scale=0.01, ortho=True, **kwargs):
    """
    Random weights drawn from a Gaussian
    """
    if len(shape) == 2:
       nin,nout = shape[0], shape[1]
    elif len(shape) == 1:
       nin = shape[0]
       nout = 1
       
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight((nin,nin))
    else:
        W = scale * np.random.randn(nin, nout)
    if len(shape) == 1:
        W = np.reshape(W, shape)
    return W.astype('float32')
    
def uniform(shape, scale=0.05, name=None,symbolic=True, **kwargs):
    if symbolic:
       return K.variable(np.random.uniform(low=-scale, high=scale, size=shape),
                      name=name)
    else:
       return np.random.uniform(low=-scale, high=scale, size=shape)

def normal(shape, scale=0.05, name=None,symbolic=True,**kwargs):
    if symbolic:
       return K.variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                      name=name)
    else:
       return np.random.normal(loc=0.0, scale=scale, size=shape)

def lecun_uniform(shape, name=None,symbolic=True, dim_ordering='th',**kwargs):
    ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale,symbolic=symbolic,name=name)


def glorot_normal(shape, name=None,symbolic=True, dim_ordering='th',**kwargs):
    ''' Reference: Glorot & Bengio, AISTATS 2010
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s, symbolic = symbolic, name=name)


def glorot_uniform(shape, name=None, symbolic=True,dim_ordering='th',**kwargs):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, symbolic=symbolic,name=name)


def he_normal(shape, name=None, symbolic=True,dim_ordering='th',**kwargs):
    ''' Reference:  He et al., http://arxiv.org/abs/1502.01852
    '''
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s,symbolic=symbolic, name=name)


def he_uniform(shape, name=None,symbolic=True, dim_ordering='th',**kwargs):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s, symbolic=symbolic,name=name)

def get_orthogonal(shape, scale=1.):
    # get the orthogonal matrix for two dim shape
    a = np.random.normal(0.0, 1.0, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.reshape(shape)
    return scale*q


def orthogonal(shape, scale=1., name=None, symbolic=True,**kwargs):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    if symbolic:
       return K.variable(scale * q[:shape[0], :shape[1]], name=name)
    else:
       return scale * q[:shape[0], :shape[1]]


def identity(shape, scale=1, dim_ordering='tf',name=None,symbolic=True,**kwargs):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise Exception('Identity matrix initialization can only be used '
                        'for 2D square matrices.')
    else:
        if symbolic:
           return K.variable(scale * np.identity(shape[0]), name=name)  
        else:
           return scale * np.identity(shape[0])

def zero(shape, name=None,symbolic=True,**kwargs):
    if symbolic:
       return K.variable(np.zeros(shape), name=name)
    else:
       return np.zeros(shape) 


def one(shape, name=None,symbolic=True,**kwargs):
    if symbolic:
       return K.variable(np.ones(shape), name=name)
    else:
       return np.ones(shape) 
from fractions import gcd
# TH kernel shape: (depth, input_depth, ...)
# TF kernel shape: (..., input_depth, depth)
def conv_init(shape, scale=1, dim_ordering='th',mode ='orthogonal', **kwargs):
    if dim_ordering == 'th':
        fout, fin, row, col = shape
    elif dim_ordering == 'tf':
        row,  col, fin, fout = shape
    print row,  col, fin, fout
    if fin == fout:
       real_data = np.zeros(shape)
    else:
       real_data = np.random.normal(0.0, 1.0, shape)    
    if dim_ordering == 'tf':
        real_data = np.transpose(real_data, (3,2,0,1))
        
    piv_r, piv_c = row//2, col//2
    if fin == fout:
       if mode == 'identity':
          real_data[:,:,piv_r, piv_c] = np.identity(fin)
       elif mode == 'orthogonal':
          a = np.random.normal(0.0, 1.0, (fin, fout))  
          u, _, v = np.linalg.svd(a, full_matrices=False)
          real_data[:,:,piv_r, piv_c] = u
       
    else:
       c= gcd(fin, fout)
       for a in range(fin):
            for b in range(fout):
                if (a*c)//fin == (b*c)//fout:
                    real_data[:,:,piv_r, piv_c] = float(c)/fout
      
    if dim_ordering == 'tf':
        res = np.transpose(real_data, (2,3,1,0))
    else:
        res = real_data
    return res


def conv_identity(shape, dim_ordering='th', name=None, symbolic=True,**kwargs):
    if len(shape) != 4:
       raise Exception("filter dimension not equal 4 is not supported yet!")
    this_data = conv_init(shape, dim_ordering=dim_ordering,mode ='identity')
    if symbolic:
      return K.variable(this_data, name=name)
    else:
      return this_data

def conv_orthogonal(shape,dim_ordering='th', name=None,symbolic=True, **kwargs):
    if len(shape) != 4:
       raise Exception("filter dimension not equal 4 is not supported yet!")
    this_data = conv_init(shape, dim_ordering=dim_ordering,mode ='orthogonal')
    if symbolic:
      return K.variable(this_data, name=name)
    else:
      return this_data

from utils.generic_utils import get_from_module
def get(identifier, **kwargs):
    return get_from_module(identifier, globals(),
                           'initialization', kwargs=kwargs)
