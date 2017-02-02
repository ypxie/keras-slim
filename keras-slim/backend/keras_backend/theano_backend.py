import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.tensor.nnet import conv3d2d
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign
import inspect
import numpy as np
from .common import _FLOATX, _EPSILON
import warnings

# INTERNAL UTILS
theano.config.floatX = _FLOATX
#_LEARNING_PHASE = theano.shared(1.0, name='keras_learning_phase')  # 0 = test, 1 = train
_LEARNING_PHASE = T.scalar(dtype='uint8',name='keras_learning_phase')  # 0 = test, 1 = train

Variable = theano.Variable  #denote the tensor variable type
FLOATX = _FLOATX

def addAttribute(x, attr=None, value=None):
    if attr is not None:
       setattr(x, attr, value)
    return x

def add_keras_shape(x, keras_shape = None):
    x._keras_shape = keras_shape
    return x
    
def learning_phase():
    # False = test, True = train
    return _LEARNING_PHASE

# VARIABLE MANIPULATION

def variable(value, dtype=_FLOATX, name=None):
    '''Instantiate a tensor variable.
    '''
    broadcastable = getattr(value, 'broadcastable', None)
    value = np.asarray(value, dtype=dtype)
    #value = value.asdtype(dtype)
    if broadcastable is not None:
        vv = theano.shared(value=value, name=name, strict=False, broadcastable= broadcastable)
    else:
        vv = theano.shared(value=value, name=name, strict=False)
    vv._keras_shape = value.shape
    return vv


def placeholder(shape=None, ndim=None, dtype=_FLOATX, name=None):
    '''Instantiate an input data placeholder variable.
    '''
    if shape is None and ndim is None:
        raise Exception('Specify either a shape or ndim value.')
    if shape is not None:
        ndim = len(shape)
    else:
        shape = tuple([None for _ in range(ndim)])

    broadcast = (False,) * ndim
    x = T.TensorType(dtype, broadcast)(name)
    x._keras_shape = shape
    x._uses_learning_phase = False
    return x


def shape(x):
    '''Return the shape of a tensor.

    Warning: type returned will be different for
    Theano backend (Theano tensor type) and TF backend (TF TensorShape).
    '''
    return x.shape

# def int_shape(x):
#     '''Returns the shape of a tensor as a tuple of
#     integers or None entries.
#     '''
#     shape = x.get_shape()
#     return tuple([i.__int__() for i in shape])

def ndim(x):
    return x.ndim


def dtype(x):
    return x.dtype


def eval(x):
    '''Run a graph.
    '''
    return x.eval()

def _none_or_int(shape):
    o = True
    for s in shape:
        if not isinstance(s, int) and s is not None:
            o = False
            break
    return o

def zeros(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-zeros variable.
    '''
    output = T.zeros(shape,  dtype=dtype)
    output.name = name
    if _none_or_int(shape):
        output._keras_shape = shape
    return output
    
def np_zeros(shape, dtype=_FLOATX, name=None):
    output = variable(np.zeros(shape), dtype, name)
    output._keras_shape = shape
    return output    

def ones(shape, dtype=_FLOATX, name=None):
    '''Instantiate an all-ones variable.
    '''
    output = T.ones(shape,  dtype=dtype)
    output.name = name
    if _none_or_int(shape):
        output._keras_shape = shape
    return output
    
def np_ones(shape, dtype=_FLOATX, name=None):
    output = variable(np.ones(shape), dtype, name)
    output._keras_shape = shape
    return output


def eye(size, dtype=_FLOATX, name=None):
    '''Instantiate an identity matrix.
    '''
    shape = (size,size)
    output = T.eye(size, dtype=dtype)
    output.name = name
    if _none_or_int(shape):
        output._keras_shape = shape
    return output
 
def np_eye(size, dtype=_FLOATX, name=None):   
    output = variable(np.eye(size), dtype, name)
    output._keras_shape = (size, size)
    return output


def ones_like(x):
    output = T.ones_like(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def zeros_like(x):
    output = T.zeros_like(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def count_params(x):
    '''Return number of scalars in a tensor.

    Return: numpy integer.
    '''
    return np.prod(x.shape.eval())


def cast(x, dtype):
    return T.cast(x, dtype)


# LINEAR ALGEBRA

'''
Assumed overridden:
+, -, /, *, +=, -=, *=, /=
'''


def dot(x, y):
    output = T.dot(x, y)
    if hasattr(x, '_keras_shape'):
        x_keras_shape = list(x._keras_shape)
    if hasattr(y, '_keras_shape'):
        y_keras_shape = list(y._keras_shape)
    if hasattr(x, '_keras_shape')  and hasattr(y, '_keras_shape'):
        if len(x_keras_shape) >= 2 and len(y_keras_shape) >= 2:
            x_keras_shape.pop(-1)
            y_keras_shape.pop(-2)
            output._keras_shape = tuple(x_keras_shape + y_keras_shape)
        elif len(x_keras_shape) == 1 and len(y_keras_shape) == 1:
            output._keras_shape = ()
        else:
            if len(x_keras_shape) > 1 and len(y_keras_shape) == 1:
                x_keras_shape.pop(-1)
                output._keras_shape = tuple(x_keras_shape)
            elif len(x_keras_shape) == 1 and len(y_keras_shape) > 1:
                raise Exception('_keras_shape not defined for dot product of \
                             input with first one-dimension and second larger than one, please fix this')
    return output


def batch_dot(x, y, axes=None):
    '''batchwise dot product
    batch_dot results in a tensor with less dimensions than the input.
    If the number of dimensions is reduced to 1, we use `expand_dims` to
    make sure that ndim is at least 2.

    # Example
        Assume x = [[1, 2]   and y = [[5, 6]
                    [3, 4]]           [7, 8]]
        batch_dot(x, y, axes=1) = [[17, 53]] which is the main diagonal
        of x.dot(y.T), although we never have to calculate the off-diagonal
        elements.


    # Arguments
        x, y: tensors with ndim >= 2
        axes: list (or single) int with target dimensions

    # Returns
        Tensor with ndim >= 2
    '''
    if type(axes) == int:
        axes = (axes, axes)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x.ndim - 1, y.ndim - 2]
    out = T.batched_tensordot(x, y, axes=axes)
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


def transpose(x):
    return T.transpose(x)


def gather(reference, indices):
    '''reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    '''
    return reference[indices]


# ELEMENT-WISE OPERATIONS


def max(x, axis=None, keepdims=False):
    return T.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return T.min(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    '''Sum of the values in a tensor, alongside the specified axis.
    '''
    output = T.sum(x, axis=axis, keepdims=keepdims)
    if hasattr(x, '_keras_shape'):
        old_keras_shape = list(x._keras_shape)
        if keepdims:
            old_keras_shape[axis] = 1
        else:
            old_keras_shape.pop(axis)
        output._keras_shape = tuple(old_keras_shape)

    return output


def prod(x, axis=None, keepdims=False):
    '''Multiply the values in a tensor, alongside the specified axis.
    '''
    output = T.prod(x, axis=axis, keepdims=keepdims)
    if hasattr(x, '_keras_shape'):
        old_keras_shape = list(x._keras_shape)
        if keepdims:
            old_keras_shape[axis] = 1
        else:
            old_keras_shape.pop(axis)
        output._keras_shape = tuple(old_keras_shape)

    return output


def mean(x, axis=None, keepdims=False):
    dtype = None
    if 'int' in x.dtype:
        dtype = _FLOATX
    output = T.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)
    if hasattr(x, '_keras_shape'):
        old_keras_shape = list(x._keras_shape)
        if keepdims:
            old_keras_shape[axis] = 1
        else:
            old_keras_shape.pop(axis)
        output._keras_shape = tuple(old_keras_shape)
    return output



def std(x, axis=None, keepdims=False):
    output = T.std(x, axis=axis, keepdims=keepdims)
    if hasattr(x, '_keras_shape'):
        old_keras_shape = list(x._keras_shape)
        if keepdims:
            old_keras_shape[axis] = 1
        else:
            old_keras_shape.pop(axis)
        output._keras_shape = tuple(old_keras_shape)

    return output


def var(x, axis=None, keepdims=False):
    output = T.var(x, axis=axis, keepdims=keepdims)
    if hasattr(x, '_keras_shape'):
        old_keras_shape = list(x._keras_shape)
        if keepdims:
            old_keras_shape[axis] = 1
        else:
            old_keras_shape.pop(axis)
        output._keras_shape = tuple(old_keras_shape)

    return output


def any(x, axis=None, keepdims=False):
    '''Bitwise reduction (logical OR).
    '''
    return T.any(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    return T.argmax(x, axis=axis, keepdims=False)


def argmin(x, axis=-1):
    return T.argmin(x, axis=axis, keepdims=False)


def square(x):
    output = T.sqr(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def abs(x):
    output = T.abs_(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def sqrt(x):
    x = T.clip(x, 0., np.inf)
    output = T.sqrt(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def exp(x):
    output = T.exp(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def log(x):
    output = T.log(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def round(x):
    output = T.round(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output

def sign(x):
    output = T.sgn(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output



def pow(x, a):
    output = T.pow(x, a)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output

def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    output = T.clip(x, min_value, max_value)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def equal(x, y):
    output = T.eq(x, y)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def not_equal(x, y):
    output = T.neq(x, y)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def maximum(x, y):
    output = T.maximum(x, y)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def minimum(x, y):
    output = T.minimum(x, y)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def sin(x):
    output = T.sin(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def cos(x):
    output = T.cos(x)
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


# SHAPE OPERATIONS

def concatenate(tensors, axis=-1):
    output = T.concatenate(tensors, axis=axis)
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


def reshape(x, shape):
    '''If you use reshape, you better specify the _keras_shape by yourself.'''
    return T.reshape(x, shape)

def permute_dimensions(x, pattern):
    '''Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    '''
    pattern = tuple(pattern)
    output = x.dimshuffle(pattern)
    if hasattr(x, '_keras_shape'):
        old_ks = x._keras_shape
        new_ks = list(old_ks)
        for idx, ind in enumerate(pattern):
            new_ks[idx] = old_ks[ind]
        output._keras_shape = tuple(new_ks)
    return output
    
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
    return T.repeat(x, rep, axis=axis)


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


def resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering):
    '''Resize the volume contained in a 5D tensor of shape
    - [batch, channels, depth, height, width] (for 'th' dim_ordering)
    - [batch, depth, height, width, channels] (for 'tf' dim_ordering)
    by a factor of (depth_factor, height_factor, width_factor).
    Both factors should be positive integers.
    '''
    if dim_ordering == 'th':
        output = repeat_elements(X, depth_factor, axis=2)
        output = repeat_elements(output, height_factor, axis=3)
        output = repeat_elements(output, width_factor, axis=4)
        return output
    elif dim_ordering == 'tf':
        output = repeat_elements(X, depth_factor, axis=1)
        output = repeat_elements(output, height_factor, axis=2)
        output = repeat_elements(output, width_factor, axis=3)
        return output
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)


def repeat(x, n):
    '''Repeat a 2D tensor.

    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    '''
    assert x.ndim == 2
    x = x.dimshuffle((0, 'x', 1))
    return T.extra_ops.repeat(x, n, axis=1)


def tile(x, n):
    return T.tile(x, n)


def flatten(x):
    return T.flatten(x)


def batch_flatten(x):
    '''Turn a n-D tensor into a 2D tensor where
    the first dimension is conserved.
    '''
    x = T.reshape(x, (x.shape[0], T.prod(x.shape) // x.shape[0]))
    return x


def expand_dims(x, dim=-1,broadcastable = True):
    '''Add a 1-sized dimension at index "dim".
    '''
    pattern = [i for i in range(x.type.ndim)]
    if dim < 0:
        if x.type.ndim == 0:
            dim = 0
        else:
            dim = dim % x.type.ndim + 1
    pattern.insert(dim, 'x')
    output = x.dimshuffle(pattern)
    
    if broadcastable != True:
       output =  T.unbroadcast(output, dim)
    if hasattr(x,'_keras_shape'):
        old_ks = list(x._keras_shape)
        old_ks.insert(dim, 1)
        output._keras_shape = tuple(old_ks)
        
    return output


def squeeze(x, axis):
    '''Remove a 1-dimension from the tensor at index "axis".
    '''
    broadcastable = x.broadcastable[:axis] + x.broadcastable[axis+1:]
    x = T.patternbroadcast(x, [i == axis for i in range(x.type.ndim)])
    x = T.squeeze(x)
    x = T.patternbroadcast(x, broadcastable)
    return x


def temporal_padding(x, padding=1):
    '''Pad the middle dimension of a 3D tensor
    with "padding" zeros left and right.

    Apologies for the inane API, but Theano makes this
    really hard.
    '''
    input_shape = x.shape
    output_shape = (input_shape[0],
                    input_shape[1] + 2 * padding,
                    input_shape[2])
    output = T.zeros(output_shape)
    return T.set_subtensor(output[:, padding:x.shape[1] + padding, :], x)


def spatial_2d_padding(x, padding=(1, 1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    '''
    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + 2 * padding[0],
                        input_shape[2] + 2 * padding[1],
                        input_shape[3])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    return T.set_subtensor(output[indices], x)


def spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='th'):
    '''Pad the 2nd, 3rd and 4th dimensions of a 5D tensor
    with "padding[0]", "padding[1]" and "padding[2]" (resp.) zeros left and right.
    '''
    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1],
                        input_shape[4] + 2 * padding[2])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]),
                   slice(padding[2], input_shape[4] + padding[2]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + 2 * padding[0],
                        input_shape[2] + 2 * padding[1],
                        input_shape[3] + 2 * padding[2],
                        input_shape[4])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(padding[2], input_shape[3] + padding[2]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    return T.set_subtensor(output[indices], x)


def pack(x):
    return T.stack(*x)

# VALUE MANIPULATION


def get_value(x):
    if not hasattr(x, 'get_value'):
        raise Exception("'get_value() can only be called on a variable. " +
                        "If you have an expression instead, use eval().")
    return x.get_value()


def batch_get_value(xs):
    '''Returns the value of more than one tensor variable,
    as a list of Numpy arrays.
    '''
    return [get_value(x) for x in xs]


def set_value(x, value):
    x.set_value(np.asarray(value, dtype=x.dtype))


def batch_set_value(tuples):
    for x, value in tuples:
        x.set_value(np.asarray(value, dtype=x.dtype))


# GRAPH MANIPULATION

class Function(object):

    def __init__(self, inputs, outputs, updates=[], **kwargs):
        self.function = theano.function(inputs, outputs, updates=updates,
                                        allow_input_downcast=True,
                                        on_unused_input='warn',
                                        **kwargs)

    def __call__(self, inputs):
        assert type(inputs) in {list, tuple}
        return self.function(*inputs)


def function(inputs, outputs, updates=[], **kwargs):
    if len(kwargs) > 0:
        function_args = inspect.getargspec(theano.function)[0]
        for key in kwargs.keys():
            if key not in function_args:
                msg = "Invalid argument '%s' passed to K.function" % key
                raise ValueError(msg)
    return Function(inputs, outputs, updates=updates, **kwargs)


def gradients(loss, variables):
    return T.grad(loss, variables)


# CONTROL FLOW

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    '''Iterates over the time dimension of a tensor.

    # Arguments
        inputs: tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape (samples, ...) (no time dimension),
                new_states: list of tensors, same length and shapes
                    as 'states'.
        initial_states: tensor with shape (samples, ...) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape (samples, time),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`scan`).
        input_length: must be specified if using `unroll`.

    # Returns
        A tuple (last_output, outputs, new_states).
            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
    '''
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    if unroll:
        if input_length is None:
            raise Exception('When specifying `unroll=True`, an `input_length` '
                            'must be provided to `rnn`.')

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == ndim-1:
            mask = expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, new_states = step_function(inputs[i], states + constants)

                if len(successive_outputs) == 0:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = T.switch(mask[i], output, prev_output)
                kept_states = []
                for state, new_state in zip(states, new_states):
                    kept_states.append(T.switch(mask[i], new_state, state))
                states = kept_states

                successive_outputs.append(output)
                successive_states.append(states)

            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))
        else:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_output = step_function(inputs[0], initial_states + constants)[0] * 0
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)

            def _step(input, mask, output_tm1, *states):
                output, new_states = step_function(input, states)
                # output previous output if masked.
                output = T.switch(mask, output, output_tm1)
                return_states = []
                for state, new_state in zip(states, new_states):
                    return_states.append(T.switch(mask, new_state, state))
                return [output] + return_states

            results, _ = theano.scan(
                _step,
                sequences=[inputs, mask],
                outputs_info=[initial_output] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if type(results) is list:
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []
    else:
        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, states = step_function(inputs[i], states + constants)
                successive_outputs.append(output)
                successive_states.append(states)
            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))

        else:
            def _step(input, *states):
                output, new_states = step_function(input, states)
                return [output] + new_states

            results, _ = theano.scan(
                _step,
                sequences=inputs,
                outputs_info=[None] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if type(results) is list:
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state[-1]) for state in states]
    return last_output, outputs, states


def switch(condition, then_expression, else_expression):
    '''condition: scalar tensor.
    '''   
    
    output = T.switch(condition, then_expression, else_expression)
    if hasattr(then_expression, '_keras_shape'):
        t_ks = then_expression._keras_shape
    if hasattr(else_expression, '_keras_shape'):
        e_ks = else_expression._keras_shape
    if hasattr(then_expression, '_keras_shape') and hasattr(else_expression, '_keras_shape'):
        t_ks = then_expression._keras_shape
        e_ks = else_expression._keras_shape
        if t_ks == e_ks:
            output._keras_shape = t_ks
        else:         
            warnings.warn('keras_shape not determined at switch function!')            
    return output

def in_train_phase(x, alt): 
    output = T.switch(_LEARNING_PHASE, x, alt)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    if hasattr(alt,'_keras_shape'):
        output._keras_shape = alt._keras_shape
    output._uses_learning_phase = True
    return output


def in_test_phase(x, alt):
    output = T.switch(_LEARNING_PHASE, alt, x)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    if hasattr(alt,'_keras_shape'):
        output._keras_shape = alt._keras_shape
    output._uses_learning_phase = True
    return output


# NN OPERATIONS

def relu(x, alpha=0., max_value=None):
    assert hasattr(T.nnet, 'relu'), ('It looks like like your version of '
                                     'Theano is out of date. '
                                     'Install the latest version with:\n'
                                     'pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps')
    output = T.nnet.relu(x, alpha)
    if max_value is not None:
        output = T.minimum(output, max_value)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    return output
    

def softmax(x):
    shape = x.shape
    x_2d = T.reshape(x, (-1, shape[-1]))

    output = T.nnet.softmax(x_2d)
    output = T.reshape(output, shape)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def softplus(x):
    output = T.nnet.softplus(x)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def softsign(x):
    output = T_softsign(x)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.categorical_crossentropy(output, target)


def sparse_categorical_crossentropy(output, target, from_logits=False):
    target = T.cast(T.flatten(target), 'int32')
    target = T.extra_ops.to_one_hot(target, nb_class=output.shape[-1])
    target = reshape(target, shape(output))
    return categorical_crossentropy(output, target, from_logits)


def binary_crossentropy(output, target, from_logits=False):
    if from_logits:
        output = T.nnet.sigmoid(output)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.binary_crossentropy(output, target)


def sigmoid(x):
    output = T.nnet.sigmoid(x)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    return output

def hard_sigmoid(x):
    output = T.nnet.hard_sigmoid(x)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    return output
    

def tanh(x):
    output = T.tanh(x)
    if hasattr(x,'_keras_shape'):
        output._keras_shape = x._keras_shape
    return output
    

def dropout(x, level, seed=None):
    if level < 0. or level >= 1:
        raise Exception('Dropout level must be in interval [0, 1[.')
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    retain_prob = 1. - level
    output  = x* rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
    output /= retain_prob
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    output = x / norm
    if hasattr(x, '_keras_shape'):
        output._keras_shape = x._keras_shape
    return output


# CONVOLUTIONS
def conv2d(x, kernel, strides=(1, 1), border_mode='valid',
           dim_ordering=_IMAGE_DIM_ORDERING, image_shape=None,
           filter_shape=None, filter_dilation=(1, 1)):
    '''2D convolution.
    # Arguments
        kernel: kernel tensor.
        strides: strides tuple.
        border_mode: string, "same" or "valid".
        dim_ordering: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    x = _preprocess_conv2d_input(x, dim_ordering)
    kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
    th_border_mode = _preprocess_border_mode(border_mode)
    np_kernel = kernel.eval()
    image_shape = _preprocess_conv2d_image_shape(dim_ordering, image_shape)
    filter_shape = _preprocess_conv2d_filter_shape(dim_ordering, filter_shape)
    #TODO: remove the if statement when theano with no filter dilation is deprecated.
    if filter_dilation == (1, 1):
        conv_out = T.nnet.conv2d(x, kernel,
                                 border_mode=th_border_mode,
                                 subsample=strides,
                                 input_shape=image_shape,
                                 filter_shape=filter_shape)
    else:
        conv_out = T.nnet.conv2d(x, kernel,
                                 border_mode=th_border_mode,
                                 subsample=strides,
                                 input_shape=image_shape,
                                 filter_shape=filter_shape,
                                 filter_dilation=filter_dilation)

    conv_out = _postprocess_conv2d_output(conv_out, x, border_mode, np_kernel,
                                          strides, dim_ordering)
    return conv_out


def in_top_k(predictions, targets, k):
    '''Says whether the `targets` are in the top `k` `predictions`

    # Arguments
        predictions: A tensor of shape batch_size x classess and type float32.
        targets: A tensor of shape batch_size and type int32 or int64.
        k: An int, number of top elements to consider.

    # Returns
        A tensor of shape batch_size and type int. output_i is 1 if
        targets_i is within top-k values of predictions_i
    '''
    predictions_top_k = T.argsort(predictions)[:, -k:]
    result, _ = theano.map(lambda prediction, target: any(equal(prediction, target)), sequences=[predictions_top_k, targets])
    return result


# CONVOLUTIONS

def _preprocess_conv2d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        x = x.dimshuffle((0, 3, 1, 2))
    return x


def _preprocess_conv3d_input(x, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols, slices)
        # TF input shape: (samples, rows, cols, slices, input_depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
    return x


def _preprocess_conv2d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        kernel = kernel.dimshuffle((3, 2, 0, 1))
    return kernel


def _preprocess_conv3d_kernel(kernel, dim_ordering):
    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH kernel shape: (depth, input_depth, rows, cols, slices)
        # TF kernel shape: (rows, cols, slices, input_depth, depth)
        kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
    return kernel


def _preprocess_border_mode(border_mode):
    if border_mode == 'same':
        th_border_mode = 'half'
    elif border_mode == 'valid':
        th_border_mode = 'valid'
    else:
        raise Exception('Border mode not supported: ' + str(border_mode))
    return th_border_mode


def _preprocess_conv2d_image_shape(dim_ordering, image_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if image_shape:
            image_shape = (image_shape[0], image_shape[3],
                           image_shape[1], image_shape[2])
    if image_shape is not None:
        image_shape = tuple(int_or_none(v) for v in image_shape)
    return image_shape


def _preprocess_conv3d_volume_shape(dim_ordering, volume_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if volume_shape:
            volume_shape = (volume_shape[0], volume_shape[4],
                            volume_shape[1], volume_shape[2], volume_shape[3])
    if volume_shape is not None:
        volume_shape = tuple(int_or_none(v) for v in volume_shape)
    return volume_shape


def _preprocess_conv2d_filter_shape(dim_ordering, filter_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if filter_shape:
            filter_shape = (filter_shape[3], filter_shape[2],
                            filter_shape[0], filter_shape[1])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _preprocess_conv3d_filter_shape(dim_ordering, filter_shape):
    # Theano might not accept long type
    def int_or_none(value):
        try:
            return int(value)
        except TypeError:
            return None
    if dim_ordering == 'tf':
        if filter_shape:
            filter_shape = (filter_shape[4], filter_shape[3],
                            filter_shape[0], filter_shape[1], filter_shape[2])
    if filter_shape is not None:
        filter_shape = tuple(int_or_none(v) for v in filter_shape)
    return filter_shape


def _postprocess_conv2d_output(conv_out, x, border_mode, np_kernel, strides, dim_ordering):
    if border_mode == 'same':
        if np_kernel.shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
        if np_kernel.shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 1))
    return conv_out


def _postprocess_conv3d_output(conv_out, x, border_mode, np_kernel, strides, dim_ordering):
    if border_mode == 'same':
        if np_kernel.shape[2] % 2 == 0:
            conv_out = conv_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :, :]
        if np_kernel.shape[3] % 2 == 0:
            conv_out = conv_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1], :]
        if np_kernel.shape[4] % 2 == 0:
            conv_out = conv_out[:, :, :, :, :(x.shape[4] + strides[2] - 1) // strides[2]]
    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 4, 1))
    return conv_out



def conv3d(x, kernel, strides=(1, 1, 1),
           border_mode='valid', dim_ordering='th',
           volume_shape=None, filter_shape=None,**kwargs):
    '''
    Run on cuDNN if available.
    border_mode: string, "same" or "valid".
    '''
    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if border_mode not in {'same', 'valid'}:
        raise Exception('Invalid border mode: ' + str(border_mode))

    if dim_ordering == 'tf':
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, conv_dim1, conv_dim2, conv_dim3)
        # TF input shape: (samples, conv_dim1, conv_dim2, conv_dim3, input_depth)
        # TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        # TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
        x = x.dimshuffle((0, 4, 1, 2, 3))
        kernel = kernel.dimshuffle((4, 3, 0, 1, 2))
        if volume_shape:
            volume_shape = (volume_shape[0], volume_shape[4],
                            volume_shape[1], volume_shape[2], volume_shape[3])
        if filter_shape:
            filter_shape = (filter_shape[4], filter_shape[3],
                            filter_shape[0], filter_shape[1], filter_shape[2])

    if border_mode == 'same':
        assert(strides == (1, 1, 1))
        pad_dim1 = (kernel.shape[2] - 1)
        pad_dim2 = (kernel.shape[3] - 1)
        pad_dim3 = (kernel.shape[4] - 1)
        output_shape = (x.shape[0], x.shape[1],
                        x.shape[2] + pad_dim1,
                        x.shape[3] + pad_dim2,
                        x.shape[4] + pad_dim3)
        output = T.zeros(output_shape)
        indices = (slice(None), slice(None),
                   slice(pad_dim1 // 2, x.shape[2] + pad_dim1 // 2),
                   slice(pad_dim2 // 2, x.shape[3] + pad_dim2 // 2),
                   slice(pad_dim3 // 2, x.shape[4] + pad_dim3 // 2))
        x = T.set_subtensor(output[indices], x)
        border_mode = 'valid'

    border_mode_3d = (border_mode, border_mode, border_mode)
    conv_out = conv3d2d.conv3d(signals=x.dimshuffle(0, 2, 1, 3, 4),
                               filters=kernel.dimshuffle(0, 2, 1, 3, 4),
                               border_mode=border_mode_3d)
    conv_out = conv_out.dimshuffle(0, 2, 1, 3, 4)

    # support strides by manually slicing the output
    if strides != (1, 1, 1):
        conv_out = conv_out[:, :, ::strides[0], ::strides[1], ::strides[2]]

    if dim_ordering == 'tf':
        conv_out = conv_out.dimshuffle((0, 2, 3, 4, 1))

    return conv_out


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
        x = x.dimshuffle((0, 3, 1, 2))

    if pool_mode == 'max':
        pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
                                ignore_border=True,
                                padding=padding,
                                mode='max')
    elif pool_mode == 'avg':
        pool_out = pool.pool_2d(x, ds=pool_size, st=strides,
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
        pool_out = pool_out.dimshuffle((0, 2, 3, 1))
    return pool_out


def pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid',
           dim_ordering='th', pool_mode='max'):
    if border_mode == 'same':
        # TODO: add implementation for border_mode="same"
        raise Exception('border_mode="same" not supported with Theano.')
    elif border_mode == 'valid':
        ignore_border = True
        padding = (0, 0)
    else:
        raise Exception('Invalid border mode: ' + str(border_mode))

    if dim_ordering not in {'th', 'tf'}:
        raise Exception('Unknown dim_ordering ' + str(dim_ordering))

    if dim_ordering == 'tf':
        x = x.dimshuffle((0, 4, 1, 2, 3))

    if pool_mode == 'max':
        # pooling over conv_dim2, conv_dim1 (last two channels)
        output = pool.pool_2d(input=x.dimshuffle(0, 1, 4, 3, 2),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=ignore_border,
                              padding=padding,
                              mode='max')

        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=ignore_border,
                                padding=padding,
                                mode='max')

    elif pool_mode == 'avg':
        # pooling over conv_dim2, conv_dim1 (last two channels)
        output = pool.pool_2d(input=x.dimshuffle(0, 1, 4, 3, 2),
                              ds=(pool_size[1], pool_size[0]),
                              st=(strides[1], strides[0]),
                              ignore_border=ignore_border,
                              padding=padding,
                              mode='average_exc_pad')

        # pooling over conv_dim3
        pool_out = pool.pool_2d(input=output.dimshuffle(0, 1, 4, 3, 2),
                                ds=(1, pool_size[2]),
                                st=(1, strides[2]),
                                ignore_border=ignore_border,
                                padding=padding,
                                mode='average_exc_pad')
    else:
        raise Exception('Invalid pooling mode: ' + str(pool_mode))

    if dim_ordering == 'tf':
        pool_out = pool_out.dimshuffle((0, 2, 3, 4, 1))
    return pool_out

# Padding
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

        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(cropping[0], output_shape[2] + cropping[0]),
                   slice(cropping[1], output_shape[3] + cropping[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1]  -     cropping[0] -  cropping[2],
                        input_shape[2]  -     cropping[1] -  cropping[3],
                        input_shape[3])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(cropping[0], output_shape[1] + cropping[0]),
                   slice(cropping[1], output_shape[2] + cropping[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    return T.set_subtensor(output[0::,0::,0::,0::], x[indices])


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
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + padding[0] + padding[2],
                        input_shape[2] + padding[1] + padding[3],
                        input_shape[3])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)

    #rex =   theano.printing.Print("T subtensoor")(T.set_subtensor(output[indices], x))
    return T.set_subtensor(output[indices], x)

def spatial_3d_padding_6specify(x, padding=(1, 1,1,1,1,1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    padding[0], and padding[1] are for left part for row and cols
    padding[2]  and padding[3] are for right part for row and col
    '''
    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] +     padding[0] + padding[3],
                        input_shape[3] +     padding[1] + padding[4],
                        input_shape[4] +     padding[2] + padding[5],
                        )
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]),
                   slice(padding[2], input_shape[4] + padding[2]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + padding[0] + padding[3],
                        input_shape[2] + padding[1] + padding[4],
                        input_shape[3] + padding[2] + padding[5],
                        input_shape[4])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(padding[2], input_shape[3] + padding[2]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    #rex =   theano.printing.Print("T subtensoor")(T.set_subtensor(output[indices], x))
    return T.set_subtensor(output[indices], x)
        

        # RANDOMNESS
        # TF uses the last dimension as channel dimension,
        # instead of the 2nd one.
        # TH input shape: (samples, input_depth, rows, cols)
        # TF input shape: (samples, rows, cols, input_depth)
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        

def lrn(x, alpha = 1e-4, k = 2, beta=0.75, n =5,dim_ordering = 'th', **kwargs):
    input_sqr = T.sqr(x)
    half_n = n // 2

    if dim_ordering == 'tf':
        x = T.tranpose(x, (0,3,1,2))
    input_shape = x.shape
    b, ch, r, c = input_shape
    extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
    input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                input_sqr)
    scale = k
    for i in range(n):
        scale += alpha * input_sqr[:,i:i+ch,:,:]
    scale = scale ** beta
    res= x / scale

    if dim_ordering == 'th':
        return res
    elif dim_ordering == 'tf':
        res = T.tranpose(res, (0,2,3,1))
        return res
    else:
        raise Exception('Unknown dim_ordering: ' + str(dim_ordering))
def random_normal(shape, mean=0.0, std=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.normal(size=shape, avg=mean, std=std, dtype=dtype)


def random_uniform(shape, low=0.0, high=1.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.uniform(shape, low=low, high=high, dtype=dtype)


def random_binomial(shape, p=0.0, dtype=_FLOATX, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    return rng.binomial(shape, p=p, dtype=dtype)
