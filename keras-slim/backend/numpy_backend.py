from numpy import *
import numpy as np

import logging
from collections import OrderedDict
from .scan_utils import *
from .keras_backend.common  import *
import warnings
_logger = logging.getLogger('theano.scan_module.scan')
_FLOATX = 'float32'
floatX = _FLOATX
dimshuffle = np.transpose
_LEARNING_PHASE = np.zeros((1))
_EPSILON = 1e-9

abs = np.absolute
class npwrapper(np.ndarray):
    '''usage: to append trainable attr to numpy object in layer initialization
       eg: b = npwrapper(np.arange(5), trainable=False) '''
    def __new__(cls, input_array, trainable=True, broadcastable=None):
        if broadcastable is None:
            broadcastable = getattr(input_array, 'broadcastable', None)

        obj = np.asarray(input_array).view(cls)
        obj.trainable = trainable
        obj._keras_shape = obj.shape
        obj.broadcastable = broadcastable
        #obj.random = np.random
        #obj.linalg = np.linalg
        return obj 

    def __array_finalize__(self, obj):
        if obj is None: return
        self.trainable = getattr(obj, 'trainable', None)
        self.set_value = getattr(obj, 'set_value', self.set_value)
        self._keras_shape = self.shape
        #self.linalg = getattr(obj, 'linalg', np.linalg)
        
    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)
    def set_value(self, value=None):
        if self.ndim == 0:
            self.fill(value)
        else:
            self[:] = value
        
def add_keras_shape(x, keras_shape = None):
    x = npwrapper(x)
    x._keras_shape = keras_shape
    return x
    
def learning_phase():
    # False = test, True = train
    return _LEARNING_PHASE

def in_train_phase(x, alt):
    x = switch(_LEARNING_PHASE, x, alt)
    if not isinstance(x, npwrapper):
       x = npwrapper(x)
    x._uses_learning_phase = True
    return x

def isnan(x):
    
    if isinstance(x, np.ndarray):
        return np.all(x==None)
    else:
        return x==None
def variable(value, dtype=_FLOATX, name=None):
    '''Instantiate a tensor variable.
    '''
    value = npwrapper(np.asarray(value, dtype=dtype))
    return value
shared = variable

def scalar(name=None, dtype=_FLOATX):
    return variable(np.asscalar(0,dtype=dtype))
    
def gradients(cost, wrt=None, **kwargs):
    '''Pretending to do gradient but actually do nothing. :)
    '''
    return wrt
grad = gradients
def function(inputlist, outputlist,**kwargs):
    if not isinstance(inputlist,list):
        inputlist = [inputlist]
    if not isinstance(outputlist,list):
        outputlist = [outputlist]
        
    def f(*inputlist, **kwargs):
        if len(outputlist) == 1:
            return outputlist[0]
        else:
            return outputlist
    return f


def sign(x):
    return T.sgn(x)


def pow(x, a):
    return np.power(x, a)


def get_value(p):
    return variable(p)

def batch_get_value(xs):
    '''Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    '''
    return [get_value(x) for x in xs]

def set_value(x, value):
    x[:] = value
    return variable(x)
    
def concatenate(tensors, axis=-1):
    output = npwrapper(np.concatenate(tensors, axis=axis))
    if hasattr(tensors[0], '_keras_shape'):
        o_ks = list(tensors[0]._keras_shape)
        single_ks = 0
        valid = True
        for t in tensors:
            if hasattr(t, '_keras_shape'):
                this_single_ks = t._keras_shape[axis]
                if this_single_ks is not None:
                    single_ks += this_single_ks
                else:
                    single_ks = None
                    break
            else:
                valid = False
                break
        if valid:
            o_ks[axis] = single_ks
            output._keras_shape = tuple(o_ks)
    return output
    
def RandomStreams(seed = 1234):
    a = np.random
    a.seed(seed)
    return a

def normal(shape=None, mean=0.0, std=1.0, dtype=_FLOATX, seed=None,rng = None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = ()
    return rng.normal(loc =mean, scale=std, size=shape).astype(dtype)


def uniform(shape=None, low=0.0, high=1.0, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = ()
    return rng.uniform(low=low, high=high, size = shape).astype(dtype)


def binomial(shape=None, p=0.0, n =1, dtype=_FLOATX, rng=None):
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        shape = ()
    return rng.binomial(n=n, p=p, size= shape).astype(dtype)

    
def multinomial(shape=None, pvals=0.0, n =1, dtype=_FLOATX, rng=None):
    '''
    pvals: should be n1*n2*...*len(p)
    shape should be n1*n2*...
    Totally mimic theano mrg_multinomial behavior.
    '''
    pvals = np.asarray(pvals)
    if rng is None:
        seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
    if shape is None:
        #shape = ()
        shape = pvals.shape
    if len(shape) == 1:
        return rng.multinomial(size= shape,n=n, pvals=pvals.flatten()).astype(dtype)
    else:
        #res_shape = shape + (pvals.shape[-1],)
        res_shape = shape
        flat_results = np.zeros(res_shape, dtype).reshape((-1, pvals.shape[-1]))
        flat_pvals   = pvals.reshape((-1, pvals.shape[-1]))
        for ind in xrange(flat_results.shape[0]):
            if ind <= flat_pvals.shape[0]:
                this_flat_p = flat_pvals[ind]
            else:
                this_flat_p = pvals.flatten()
            flat_results[ind] = rng.multinomial(n=n, pvals= this_flat_p-1e-8).astype(dtype)
        
        results = flat_results.reshape(res_shape)
        return results

def matrix(name='x', dtype='float32', shape = None):
    x= np.zeros(shape).astype(dtype)
    return variable(x)

def tensor3(name='x', dtype='float32', shape = None):
    x= np.zeros(shape).astype(dtype)
    return variable(x)

def tensor4(name='x', dtype='float32', shape = None):
    x= np.zeros(shape).astype(dtype)
    return variable(x)

def set_subtensor(dest, source):
    # you can not use return value, since dest can be a indexed tensor which 
    # might not have a desired shape
    dest = source
    raise warnings('dest in set_subtensor is the sliced version of dest, use assign_subtensor instead')
    return dest

def assign_subtensor(dest, source, dest_slice=None):
    if dest_slice is None:
        dest[:] = source[:]
    else:
        dest[dest_slice] = source
    return dest    

def alloc(value, shape,broadcastable =True):
    a = zeros(tuple(shape)) + value
    return variable(a)

def unbroadcast(x, *axes):
	return x
def addbroadcast(x, *axes):
	return x

def expand_dims(x, dim=-1, broadcastable=True):
    '''Add a 1-sized dimension at index "dim".
    '''
    return npwrapper(np.expand_dims(x, axis=dim))

def shape_padleft(t, n_ones=1):
    pattern = [1] * n_ones + [t.shape[i] for i in xrange(t.ndim)]
    return np.reshape(t, pattern)
    
def shape_padright(t, n_ones=1):
    pattern =  [t.shape[i] for i in xrange(t.ndim)] + [1] * n_ones
    return np.reshape(t, pattern)
    
def shape_padaxis(t, axis):
    """Reshape `t` by inserting 1 at the dimension `axis`.

    Example
    -------
    >>> tensor = theano.tensor.tensor3()
    >>> theano.tensor.shape_padaxis(tensor, axis=0)
    DimShuffle{x,0,1,2}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=1)
    DimShuffle{0,x,1,2}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=3)
    DimShuffle{0,1,2,x}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=-1)
    DimShuffle{0,1,2,x}.0
    See Also
    --------
    shape_padleft
    shape_padright
    Dimshuffle
    """
    ndim = t.ndim + 1
    if not -ndim <= axis < ndim:
        msg = 'axis {0} is out of bounds [-{1}, {1})'.format(axis, ndim)
        raise IndexError(msg)
    if axis < 0:
        axis += ndim

    pattern = [t.shape[i] for i in xrange(t.ndim)]
    pattern.insert(axis, 1)
    return np.reshape(t,pattern)
def shape(x):
    '''Return the shape of a tensor.

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).
    '''
    return x.shape
def ndim(x):
    return x.ndim


def dtype(x):
    return x.dtype


def eval(x):
    '''Run a graph.
    '''
    return x
def ones_like(x):
    return variable(ones(x.shape))


def zeros_like(x):
    return npwrapper(zeros(x.shape))

def zeros(shape,dtype= _FLOATX, name=None,**kwargs):
    return variable(np.zeros(shape), dtype=dtype, name=name)

def ones(shape, dtype= _FLOATX,name=None,**kwargs):
    return variable(np.ones(shape), dtype=dtype, name=name)

def count_params(x):
    '''Return number of scalars in a tensor.

    Return: numpy integer.
    '''
    return np.prod(x.shape)


def cast(x, dtype):
    return cast(x, dtype)

def gather(reference, indices):
    '''reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    '''
    return reference[indices]

def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return np.clip(x, min_value, max_value)

def equal(x, y):
    return np.equal(x, y)


def not_equal(x, y):
    return np.not_equal(x, y)


def permute_dimensions(x, pattern):
    '''Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    '''
    pattern = tuple(pattern)
    return x.transpose(pattern)

def reverse(x, axis=None):
    ndim = x.ndim
    slice_order = [slice(0,None,1) for _ in range(ndim)]
    slice_order[axis] = slice(-1, None,-1)
    output = x[tuple(slice_order)]
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output
    
def repeat_elements(x, rep, axis):
    '''Repeat the elements of a tensor along an axis, like np.repeat.

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3).
    '''
    return np.repeat(x, rep, axis=axis)

def resize_images(X, height_factor, width_factor, dim_ordering):
    '''Resize the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'th' dim_ordering)
    - [batch, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if dim_ordering == 'th':
        output = repeat_elements(X, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    elif dim_ordering == 'tf':
        output = repeat_elements(X, height_factor, axis=1)
        output = repeat_elements(output, width_factor, axis=2)
        return output
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)

def repeat(x, n):
    '''Repeat a 2D tensor.
    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    '''
    assert x.ndim == 2
    x = x[:,np.newaxis,:]
    return np.repeat(x, n, axis=1)
    
def batch_flatten(x):
    '''Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.
    '''
    x = np.reshape(x, (x.shape[0], np.prod(x.shape) // x.shape[0]))
    return x

def pack(x):
    return np.stack(*x)

def switch(condition, then_expression, else_expression):
    '''condition: scalar tensor.
    '''
    return variable(np.where(condition, then_expression, else_expression))

def relu(x, alpha=0., max_value=None):
    x = switch(x>0, x, alpha*x)
    if max_value is not None:
        x = np.minimum(x, max_value)
    return x
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum()
    shape = x.shape
    x_2d = np.reshape(x, (-1, shape[-1]))

    e_x = np.exp(x_2d - x_2d.max(axis=1, keepdims=True))
    out_2d = e_x / e_x.sum(axis=1, keepdims=True)
    out = np.reshape(out_2d, shape)
    return out

def hard_sigmoid(x):
    return sigmoid(x)

def softplus(x):
    return log(1+exp(x))

def softsign(x):
    return x/(1+abs(x))


def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = clip(output, _EPSILON, 1.0 - _EPSILON)
    return np.sum(-target*log(output)) 


def sparse_categorical_crossentropy(output, target, from_logits=False):
    target = T.cast(T.flatten(target), 'int32')
    target = T.extra_ops.to_one_hot(target, nb_class=output.shape[-1])
    target = reshape(target, shape(output))
    return np.sum(-target*log(output))
    


def binary_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = clip(output, _EPSILON, 1.0 - _EPSILON)
    return np.sum(- target*log(output) - ( (1-target)*log(1-output) ))




def l2_normalize(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x / norm

# CONVOLUTIONS
import autograd.scipy.signal

def np_conv2d(x, k, axes=([2, 3], [2, 3]), dot_axes = ([1], [0]),**kwargs):
    '''
    x: inputs [data, color_in, y, x]
    k: [color_out, color_in, y, x]
    '''
    k = k.transpose((1,0,2,3))
    res = autograd.scipy.signal.convolve(x,k,axes=axes,dot_axes=dot_axes, **kwargs)
    return res

# np_conv2d, the shape information is:
# Input_shape: [data, color_in, y, x]
# Params dimensions: [color_in, color_out, y, x]
# Output dimensions: [data, color_out, y, x]
def conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th',
           image_shape=None, filter_shape=None, dilated = 0, rate = 1,**kwargs):
    '''
    border_mode: string, "same" or "valid".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))
    if dilated == 0:
        rate = [rate, rate]
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        x = x.transpose((0, 3, 1, 2))
        kernel = kernel.transpose((3, 2, 0, 1))
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])
    if border_mode == 'same':
        th_border_mode = 'half'
        np_kernel = kernel
        ks = np_kernel.shape
        pad4d = ((0,0),(0,0),(ks[2]/2 -1, ks[2]-ks[2]/2), (ks[3]/2-1, ks[3]-ks[3]/2) )
        img = np.pad(x, pad4d, mode = 'constant')
        th_border_mode = 'valid'

    elif border_mode == 'valid':
         th_border_mode = 'valid'
    else:
        raise Exception('Border mode not supported: ' + str(border_mode))

    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None

    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)

    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)

    conv_out = np_conv2d(x, kernel, mode=th_border_mode)

    if border_mode == 'same':
        if np_kernel.shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
        if np_kernel.shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]

    if dim_ordering == 'tf':
        conv_out = conv_out.transpose((0, 2, 3, 1))
    conv_out = npwrapper(conv_out)
    return conv_out

def np_pool_2d(inputs, ds=(2,2), st=2, padding=(0,0), mode='max', **kwargs):
    new_shape = inputs.shape[:2]
    pad3d = ((0,0),(0,0), (padding[0], padding[0]), (padding[1], padding[1]))
    inputs = np.pad(inputs, pad3d, mode = 'constant')

    expected_width = ( (inputs.shape[2] + st[0] -1 ) // st[0]) * st[0]
    expected_height =( (inputs.shape[3] + st[1] -1) // st[1]) * st[1]

    residue = ((0,0),(0,0),(0,expected_width-inputs.shape[2] ),(0,expected_height-inputs.shape[3]))
    inputs = np.pad(inputs, residue, mode = 'constant')

    inputs = inputs[:,:,0:expected_width, 0:expected_height]

    for i in [0, 1]:
        pool_width = ds[i]
        img_width = inputs.shape[i + 2]
        new_shape += (pool_width, img_width / pool_width)
    result = inputs.reshape(new_shape)
    if mode == 'max':
        res = np.max(np.max(result, axis=2), axis=3)
    elif mode == 'avg':
        res = np.mean(np.mean(result, axis=2), axis=3)
    res = npwrapper(res)
    return res

def pool2d(x, pool_size, strides=(1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    if border_mode == 'same':
        w_pad = pool_size[0] - 2 if pool_size[0] % 2 == 1 else pool_size[0] - 1
        h_pad = pool_size[1] - 2 if pool_size[1] % 2 == 1 else pool_size[1] - 1
        padding = (w_pad, h_pad)
    elif border_mode == 'valid':
        padding = (0, 0)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        x = x.transpose((0, 3, 1, 2))

    if pool_mode == 'max':
        pool_out = np_pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=True,
                                padding=padding,
                                mode='max')
    elif pool_mode == 'avg':
        pool_out = np_pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=True,
                                padding=padding,
                                mode='average_exc_pad')
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    if border_mode == 'same':
        expected_width = (x.shape[2] + strides[0] - 1) // strides[0]
        expected_height = (x.shape[3] + strides[1] - 1) // strides[1]

        pool_out = pool_out[:, :,
                            : expected_width,
                            : expected_height]

    if dim_ordering == 'tf':
        pool_out = pool_out.transpose((0, 2, 3, 1))
    pool_out = npwrapper(pool_out)
    return pool_out


def spatial_2d_cropping_4specify(x, cropping=(1, 1, 1, 1), dim_ordering='th'):
    '''croping the 2nd and 3rd dimensions of a 4D tensor
     cropping[0], and cropping[1] are for left part for row and cols
     cropping[2]  and cropping[3] are for right part for row and col
    '''
    input_shape = list(x.shape)
    cropping = list(cropping)
    if dim_ordering == 'th':
        #input_shape[2] = theano.printing.Print("this is input_shape[2]: " )(input_shape[2])
        #cropping[0] = theano.printing.Print("this is cropping[0]: " )(cropping[0])
        #cropping[2] = theano.printing.Print("this is cropping[2]: " )(cropping[2])

        output_shape = [input_shape[0],
                        input_shape[1],
                        input_shape[2] -     cropping[0] -  cropping[2],
                        input_shape[3] -     cropping[1] -  cropping[3]]

        output = np.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(cropping[0], output_shape[2] + cropping[0]),
                   slice(cropping[1], output_shape[3] + cropping[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1]  -     cropping[0] -  cropping[2],
                        input_shape[2]  -     cropping[1] -  cropping[3],
                        input_shape[3])
        output = np.zeros(output_shape)
        indices = (slice(None),
                   slice(cropping[0], output_shape[1] + cropping[0]),
                   slice(cropping[1], output_shape[2] + cropping[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    
    output[0::,0::,0::,0::] = x[indices]
    output = npwrapper(output)
    return output


def spatial_2d_padding_4specify(x, padding=(1, 1,1,1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    padding[0], and padding[1] are for left part for row and cols
    padding[2]  and padding[3] are for right part for row and col
    '''
    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] +     padding[0] + padding[2],
                        input_shape[3] +     padding[1] + padding[3])
        output = np.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + padding[0] + padding[2],
                        input_shape[2] + padding[1] + padding[3],
                        input_shape[3])
        output = np.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    output[indices] = x
    output = npwrapper(output)
    return output
    
def scan(fn,
         sequences=None,
         outputs_info=None,
         non_sequences=None,
         n_steps=None,
         truncate_gradient=-1,
         go_backwards=False,
         mode=None,
         name=None,
         profile=False,
         allow_gc=None,
         strict=False):
    def wrap_into_list(x):
        """
        Wrap the input into a list if it is not already a list.

        """
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)

    seqs = wrap_into_list(sequences)
    outs_info = wrap_into_list(outputs_info)

    # Make sure we get rid of numpy arrays or ints or anything like that
    # passed as inputs to scan
    non_seqs = wrap_into_list(non_sequences)

    # If we provided a known number of steps ( before compilation)
    # and if that number is 1 or -1, then we can skip the Scan Op,
    # and just apply the inner function once
    # To do that we check here to see the nature of n_steps
    n_fixed_steps = None

    if isinstance(n_steps, (float, int)):
        n_fixed_steps = int(n_steps)
    else:
        if n_steps is None:
            n_fixed_steps = None
        

    # Check n_steps is an int
    if (hasattr(n_steps, 'dtype') and
        str(n_steps.dtype)[:3] not in ('uin', 'int')):
        raise ValueError(' n_steps must be an int. dtype provided '
                         'is %s' % n_steps.dtype)

    # compute number of sequences and number of outputs
    n_seqs = len(seqs)
    n_outs = len(outs_info)

    return_steps = OrderedDict()
    # wrap sequences in a dictionary if they are not already dictionaries
    for i in xrange(n_seqs):
        if not isinstance(seqs[i], dict):
            seqs[i] = OrderedDict([('input', seqs[i]), ('taps', [0])])
        elif seqs[i].get('taps', None) is not None:
            seqs[i]['taps'] = wrap_into_list(seqs[i]['taps'])
        elif seqs[i].get('taps', None) is None:
            # seqs dictionary does not have the ``taps`` key
            seqs[i]['taps'] = [0]

    # wrap outputs info in a dictionary if they are not already in one
    for i in xrange(n_outs):
        if outs_info[i] is not None:
            if isinstance(outs_info[i], dict):
                # DEPRECATED :
                if outs_info[i].get('return_steps', None) is not None:
                    raise ValueError(
                            "Using `return_steps` has been deprecated. "
                            "Simply select the entries you need using a "
                            "subtensor. Scan will optimize memory "
                            "consumption, so do not worry about that.")
                # END

            if not isinstance(outs_info[i], dict):
                # by default any output has a tap value of -1
                outs_info[i] = OrderedDict([('initial', outs_info[i]), ('taps', [-1])])
            elif (outs_info[i].get('initial', None) is None and
                    outs_info[i].get('taps', None) is not None):
                # ^ no initial state but taps provided
                raise ValueError(('If you are using slices of an output '
                                  'you need to provide a initial state '
                                  'for it'), outs_info[i])
            elif (outs_info[i].get('initial', None) is not None and
                  outs_info[i].get('taps', None) is None):
                # ^ initial state but taps not provided
                if 'taps' in outs_info[i]:
                    # ^ explicitly provided a None for taps
                    _logger.warning('Output %s ( index %d) has a initial '
                            'state but taps is explicitly set to None ',
                             getattr(outs_info[i]['initial'], 'name', 'None'),
                             i)
                outs_info[i]['taps'] = [-1]
        else:
            # if a None is provided as the output info we replace it
            # with an empty OrdereDict() to simplify handling
            outs_info[i] = OrderedDict()



    ##
    # Step 2. Generate inputs and outputs of the inner functions
    # for compiling a dummy function (Iteration #1)
    ##

    # create theano inputs for the recursive function
    # note : this is a first batch of possible inputs that will
    #        be compiled in a dummy function; we used this dummy
    #        function to detect shared variables and their updates
    #        and to construct a new and complete list of inputs and
    #        outputs

    n_seqs = 0
    scan_seqs = []     # Variables passed as inputs to the scan op
    inner_seqs = []    # Variables passed as inputs to the inner function
    inner_slices = []  # Actual slices if scan is removed from the picture
    
    actual_looping_size = np.inf # find the least looping size, this is used to allocate the size of output
    for i,seq in enumerate(seqs):
        if 'taps' in seq:
            # go through the indicated slice
            mintap = np.min(seq['taps'])
            maxtap = np.max(seq['taps'])
            
            maxtap_proxy = np.max((maxtap, 0))
            mintap_proxy = np.min((mintap, 0))

            this_length = seq['input'].shape[0] - abs(maxtap_proxy) - abs(mintap_proxy)
            actual_looping_size = np.min((actual_looping_size, this_length))

            for k in seq['taps']:           
                actual_slice = seq['input'][k - mintap]
                _seq_val = seq['input'] #tensor.as_tensor_variable(seq['input'])
                _seq_val_slice = _seq_val[k - mintap]
                nw_slice = _seq_val_slice
               
                start = (k - mintap_proxy)
                if k == maxtap_proxy:
                    nw_seq = seq['input'][start:]
                else:
                    end = -(maxtap_proxy - k)
                    nw_seq = seq['input'][start:end]

                if go_backwards:
                    nw_seq = nw_seq[::-1]

                scan_seqs.append(nw_seq)
                inner_seqs.append(nw_slice)
                inner_slices.append(actual_slice)
                n_seqs += 1
            
    # Since we've added all sequences now we need to level them up based on
    # n_steps or their different shapes
    lengths_vec = []
    for seq in scan_seqs:
        lengths_vec.append(seq.shape[0])

    if not isNaN_or_Inf_or_None(n_steps):
        # ^ N_steps should also be considered
        lengths_vec.append(n_steps)

    if len(lengths_vec) == 0:
        # ^ No information about the number of steps
        raise ValueError('No information about the number of steps '
                         'provided. Either provide a value for '
                         'n_steps argument of scan or provide an input '
                         'sequence')

    # If the user has provided the number of steps, do that regardless ( and
    # raise an error if the sequences are not long enough )
    if isNaN_or_Inf_or_None(n_steps):
        actual_n_steps = lengths_vec[0]
        for contestant in lengths_vec[1:]:
            actual_n_steps = np.minimum(actual_n_steps, contestant)
    else:
        actual_n_steps = n_steps

    # Add names -- it helps a lot when debugging

    for (nw_seq, seq) in zip(scan_seqs, seqs):
        if getattr(seq['input'], 'name', None) is not None:
            nw_seq.name = seq['input'].name + '[%d:]' % k

    scan_seqs = [seq[:actual_n_steps] for seq in scan_seqs]
    # Conventions :
    #   mit_mot = multiple input taps, multiple output taps ( only provided
    #             by the gradient function )
    #   mit_sot = multiple input taps, single output tap (t + 0)
    #   sit_sot = single input tap, single output tap (t + 0)
    #   nit_sot = no input tap, single output tap (t + 0)

    # MIT_MOT -- not provided by the user only by the grad function
    n_mit_mot = 0
    n_mit_mot_outs = 0
    mit_mot_scan_inputs = []
    mit_mot_inner_inputs = []
    mit_mot_inner_outputs = []
    mit_mot_out_slices = []
    mit_mot_rightOrder = []

    # SIT_SOT -- provided by the user
    n_mit_sot = 0
    mit_sot_scan_inputs = []
    mit_sot_inner_inputs = []
    mit_sot_inner_slices = []
    mit_sot_inner_outputs = []
    mit_sot_return_steps = OrderedDict()
    mit_sot_tap_array = []
    mit_sot_rightOrder = []

    n_sit_sot = 0
    sit_sot_scan_inputs = []
    sit_sot_inner_inputs = []
    sit_sot_inner_slices = []
    sit_sot_inner_outputs = []
    sit_sot_return_steps = OrderedDict()
    sit_sot_rightOrder = []

    # go through outputs picking up time slices as needed
    for i, init_out in enumerate(outs_info):
        # Note that our convention dictates that if an output uses
        # just the previous time step, as a initial state we will only
        # provide a tensor of the same dimension as one time step; This
        # makes code much cleaner for those who do not use taps. Otherwise
        # they would always had to shape_padleft the initial state ..
        # which is ugly
        if init_out.get('taps', None) == [-1]:

            actual_arg = init_out['initial']
            #if not isinstance(actual_arg, tensor.Variable):
            #    actual_arg = tensor.as_tensor_variable(actual_arg)
            arg = safe_new(actual_arg)


            if getattr(init_out['initial'], 'name', None) is not None:
                arg.name = init_out['initial'].name + '[t-1]'

            # We need now to allocate space for storing the output and copy
            # the initial state over. We do this using the expand function
            # defined in scan utils
            sit_sot_scan_inputs.append(
                    expand_empty(
                    unbroadcast(
                        shape_padleft(actual_arg), 0),
                    actual_n_steps
                ))

            sit_sot_inner_slices.append(actual_arg)
            if i in return_steps:
                sit_sot_return_steps[n_sit_sot] = return_steps[i]
            sit_sot_inner_inputs.append(arg)
            sit_sot_rightOrder.append(i)
            n_sit_sot += 1

        elif init_out.get('taps', None):

            if np.any(np.array(init_out.get('taps', [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                    init_out)
            # go through the taps
            mintap = abs(np.min(init_out['taps']))
            mit_sot_tap_array.append(init_out['taps'])
            idx_offset = abs(np.min(init_out['taps']))
            # Sequence
            mit_sot_scan_inputs.append(
                expand_empty(init_out['initial'][:mintap],
                                        actual_n_steps))

            if i in return_steps:
                mit_sot_return_steps[n_mit_sot] = return_steps[i]
            mit_sot_rightOrder.append(i)
            n_mit_sot += 1
            for k in init_out['taps']:
                # create a new slice
                actual_nw_slice = init_out['initial'][k + mintap]
                _init_out_var = init_out['initial']
                _init_out_var_slice = _init_out_var[k + mintap]
                nw_slice = type(_init_out_var_slice)

                # give it a name or debugging and pretty printing
                if getattr(init_out['initial'], 'name', None) is not None:
                    if k > 0:
                        nw_slice.name = (init_out['initial'].name +
                                            '[t+%d]' % k)
                    elif k == 0:
                        nw_slice.name = init_out['initial'].name + '[t]'
                    else:
                        nw_slice.name = (init_out['initial'].name +
                                            '[t%d]' % k)
                mit_sot_inner_inputs.append(nw_slice)
                mit_sot_inner_slices.append(actual_nw_slice)
        # NOTE: there is another case, in which we do not want to provide
        #      any previous value of the output to the inner function (i.e.
        #      a map); in that case we do not have to do anything ..

    # Re-order args
    max_mit_sot = np.max([-1] + mit_sot_rightOrder) + 1
    max_sit_sot = np.max([-1] + sit_sot_rightOrder) + 1
    n_elems = np.max([max_mit_sot, max_sit_sot])
    _ordered_args = [[] for x in xrange(n_elems)]
    offset = 0
    for idx in xrange(n_mit_sot):
        n_inputs = len(mit_sot_tap_array[idx])
        if n_fixed_steps in [1, -1]:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_slices[offset:offset + n_inputs]
        else:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_inputs[offset:offset + n_inputs]
        offset += n_inputs

    for idx in xrange(n_sit_sot):
        if n_fixed_steps in [1, -1]:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_slices[idx]]
        else:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_inputs[idx]]

    ordered_args = []
    for ls in _ordered_args:
        ordered_args += ls
    if n_fixed_steps in [1, -1]:
        args = (inner_slices +
                ordered_args +
                non_seqs)

    else:
        args = (inner_seqs +
                ordered_args +
                non_seqs)
 
    # add only the non-shared variables and non-constants to the arguments of
    # the dummy function [ a function should not get shared variables or
    # constants as input ]
    #dummy_args = [arg for arg in args
    #              if (not isinstance(arg, SharedVariable) and
    #                  not isinstance(arg, tensor.Constant))]
    dummy_args = [arg for arg in args]
    # when we apply the lambda expression we get a mixture of update rules
    # and outputs that needs to be separated
    
    condition, outputs, updates = get_updates_and_outputs(fn(*args))
    if condition is not None:
        as_while = True
    else:
        as_while = False
    if not (len(outputs) == n_outs or outs_info == []):
        raise ValueError('Please provide None as outputs_info for '
                         'any output that does not feed back into '
                         'scan (i.e. it behaves like a map) ')
    # now we can allocate all the memory of outputs tensor which is originally null
    if not isinstance(outputs, list):
       outputs = [outputs]
    
    OUTPUTS_AS_INPUT_IND = [0 for _ in outputs] # used to index Not_None outputs_info
    Not_None_outpus_ind = []
    for idx, each in enumerate(outs_info):
        if each.get('initial', None) is not None:
            OUTPUTS_AS_INPUT_IND[idx] = 1
            Not_None_outpus_ind.append(idx)
       
    # we add actual n step
    outputs_loop = [] 
    for i, init_out in enumerate(outs_info):

        if init_out.get('taps', None) == [-1]:

            actual_arg = init_out['initial']
            #if not isinstance(actual_arg, tensor.Variable):
            #    actual_arg = tensor.as_tensor_variable(actual_arg)
            arg = safe_new(actual_arg)

            if getattr(init_out['initial'], 'name', None) is not None:
                arg.name = init_out['initial'].name + '[t-1]'

            # We need now to allocate space for storing the output and copy
            # the initial state over. We do this using the expand function
            # defined in scan utils
            outputs_loop.append(
                    expand_empty(
                    unbroadcast(
                        shape_padleft(actual_arg), 0),
                    actual_n_steps
                ))

            #sit_sot_inner_slices.append(actual_arg)
            #if i in return_steps:
            #    sit_sot_return_steps[n_sit_sot] = return_steps[i]
            #sit_sot_inner_inputs.append(arg)
            sit_sot_rightOrder.append(i)
            n_sit_sot += 1

        elif init_out.get('taps', None):

            if np.any(np.array(init_out.get('taps', [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                    init_out)
            # go through the taps
            mintap = abs(np.min(init_out['taps']))
            #mit_sot_tap_array.append(init_out['taps'])
            idx_offset = abs(np.min(init_out['taps']))
            # Sequence
            outputs_loop.append(
                expand_empty(init_out['initial'][:mintap],
                                        actual_n_steps))
        else: # the None outputs_info, we only has actual_n_steps
            
            outputs_loop.append(
                    expand_empty(
                    unbroadcast(
                        shape_padleft(outputs[i]), 0),
                    actual_n_steps -1 
                ))

    for loop_ind in range(actual_n_steps):
        real_seq_inp = []
        real_outputs_inp = []
        for seq_inp in scan_seqs:
            real_seq_inp.append(seq_inp[loop_ind])
        for idx, outputful_tensor in enumerate(outputs_loop):
             
            init_out = outs_info[idx]
            
            if init_out.get('taps', None):
                this_taps = init_out['taps']
                min_tap = np.min(this_taps)
                max_tap = np.max(this_taps)
                this_tensor = outputful_tensor[loop_ind:loop_ind+abs(min_tap)]

                #now we have a subtensor at current cursor.

                for k in init_out['taps']:
                    current_ind = abs(min_tap) + k
                    real_outputs_inp.append(this_tensor[current_ind])
        args = (real_seq_inp +
                real_outputs_inp +
                non_seqs)
        cur_condition, cur_outputs, cur_updates = get_updates_and_outputs(fn(*args))
        #now we need to update the full outputs tensor
        for oind, cur_output in enumerate(cur_outputs):
            if OUTPUTS_AS_INPUT_IND[oind] == 1: #means it's used as inputs
               outputs_loop[oind][loop_ind+1] = cur_output
            else:
               outputs_loop[oind][loop_ind] = cur_output 
    #now we need to prepare the output
    return_outputs =[]
    for rind, o_tensor in enumerate(outputs_loop):
        if outs_info[rind].get('taps', None):
           extra_len = abs(min(outs_info[rind].get('taps', None)))
        else:
           extra_len = 0 #because we only pad n_step -1 step
        return_outputs.append(o_tensor[extra_len:]) 
 
    return return_outputs, updates
