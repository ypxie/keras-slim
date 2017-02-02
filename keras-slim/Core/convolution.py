import ..backend.export as T
from   ..backend.export import npwrapper
import numpy as np
from .frame import *
from ..utils import activations, initializations, regularizers, np_utils

def init_conv2dlayer(options, params, input_shape, nb_filter, nb_row, nb_col,prefix='conv',
	             init='glorot_uniform', dim_ordering= 'th',bias=True, trainable=True, **kwargs):

    init = initializations.get(init, dim_ordering=dim_ordering)
    if dim_ordering == 'th':
        stack_size = input_shape[1]
        W_shape = (nb_filter, stack_size, nb_row, nb_col)
    elif dim_ordering == 'tf':
        stack_size = input_shape[3]
        W_shape = (nb_row, nb_col, stack_size, nb_filter)
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
       
    W = init(W_shape, dim_ordering = dim_ordering, symbolic=False)
    b = initializations.get('zero')((nb_filter),symbolic=False)
    params[get_name(prefix, 'W')] = npwrapper(W, trainable=trainable)
    params[get_name(prefix, 'b')] = npwrapper(b, trainable=trainable)
    
    return params

def conv2dlayer(tparams,x, options, nb_filter, nb_row, nb_col, prefix='conv', 	       
            border_mode='valid', subsample=(1, 1), dim_ordering= 'th',
	        activation='linear', W_regularizer=None, 
	        b_regularizer=None, activity_regularizer=None,
            W_constraint=None, b_constraint=None, 
            dilated = 0, rate = 1,**kwargs):

    module_identifier = get_layer_identifier(prefix)
    init_LayerInfo(options, name = module_identifier)
    thismodule = options[module_identifier]
    
    
    input_shape = x._keras_shape
    def get_output_shape_for(input_shape):
        if dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)

        rows = np_utils.conv_output_length(rows, nb_row,border_mode, subsample[0])
        cols = np_utils.conv_output_length(cols, nb_col,border_mode, subsample[1])

        if dim_ordering == 'th':
            return (input_shape[0], nb_filter, rows, cols)
        elif dim_ordering == 'tf':
            return (input_shape[0], rows, cols, nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    
    activation_func = activations.get(activation) 
    if build_or_not(module_identifier, options):
        if W_regularizer:
            W_regularizer.set_param(tparams[get_name(prefix,'W')])
            thismodule.regularizers.append(W_regularizer)
        if b_regularizer:
            b_regularizer.set_param(tparams[get_name(prefix,'b')])
            thismodule.regularizers.append(b_regularizer)

    if dim_ordering == 'th':
        stack_size = input_shape[1]
        W_shape = (nb_filter, stack_size, nb_row, nb_col)
    elif dim_ordering == 'tf':
        stack_size = input_shape[3]
        W_shape = (nb_row, nb_col, stack_size, nb_filter)

    output = T.conv2d(x, tparams[get_name(prefix,'W')], strides=subsample,
                          dilated = dilated, rate= rate,
                          border_mode= border_mode,
                          dim_ordering= dim_ordering,
                          filter_shape= W_shape)
    if dim_ordering == 'th':
        output += T.reshape(tparams[get_name(prefix,'b')], (1, nb_filter, 1, 1))
    elif dim_ordering == 'tf':
        output += T.reshape(tparams[get_name(prefix,'b')], (1, 1, 1, nb_filter))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    
    output = activation_func(output)
    output._keras_shape = get_output_shape_for(input_shape)
    return output

def Convolution2D(options, nb_filter, nb_row, nb_col, prefix='conv',
                  init='glorot_uniform', border_mode='valid', subsample=(1, 1),dim_ordering= 'th',
                  activation='linear', W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                  W_constraint=None, b_constraint=None,bias=True, dilated = 0, rate = 1, trainable=True, 
                  belonging_Module=None,**kwargs):
    '''
    params > tparams > empty
    if params covers all the weights_keys. use params to update tparams.
    '''
    if belonging_Module is None:
        belonging_Module = options['belonging_Module'] if belonging_Module in options else None
   
    def f(x, tparams, options, params = None ):
        tmp_params = OrderedDict()
        module_identifier = get_layer_identifier(prefix)
        if build_or_not(module_identifier, options):
            init_LayerInfo(options, name = module_identifier)
        
        input_shape = x._keras_shape
        tmp_params = init_conv2dlayer(options, tmp_params, input_shape, nb_filter, nb_row, nb_col,prefix=prefix,
                    init=init,dim_ordering= dim_ordering,bias=bias, trainable=trainable)
        update_or_init_params(tparams, params, tmp_params=tmp_params)
        output = conv2dlayer(tparams, x, options, nb_filter, nb_row, nb_col, prefix=prefix,
                            border_mode=border_mode,subsample=subsample, dim_ordering= dim_ordering,
                            activation=activation, W_regularizer=W_regularizer, b_regularizer=b_regularizer, 
                            activity_regularizer=activity_regularizer,W_constraint=W_constraint,
                            b_constraint=b_constraint,dilated = 0, rate = 1,**kwargs)
        
        if build_or_not(module_identifier, options):
            updateModuleInfo(options, tparams, prefix, module_identifier)
            update_father_module(options,belonging_Module, module_identifier)
        return output
    
    return f
    
          
def MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same', dim_ordering='th'):
    
    def func(inputs):
        input_shape = inputs._keras_shape
        def get_output_shape_for(input_shape):
            if dim_ordering == 'th':
                rows = input_shape[2]
                cols = input_shape[3]
            elif dim_ordering == 'tf':
                rows = input_shape[1]
                cols = input_shape[2]
            else:
                raise Exception('Invalid dim_ordering: ' + dim_ordering)

            rows = np_utils.conv_output_length(rows, pool_size[0],
                                    border_mode, strides[0])
            cols = np_utils.conv_output_length(cols, pool_size[1],
                                    border_mode, strides[1])

            if dim_ordering == 'th':
                return (input_shape[0], input_shape[1], rows, cols)
            elif dim_ordering == 'tf':
                return (input_shape[0], rows, cols, input_shape[3])
            else:
                raise Exception('Invalid dim_ordering: ' + dim_ordering)
                
        output = T.pool2d(inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')

        output._keras_shape = get_output_shape_for(input_shape)
        return output
    return func

def UpSampling2D( size=(2,2),mask=None,dim_ordering = 'th'):
    
    def func(inputs):
        input_shape = inputs._keras_shape
        def get_output_shape_for(input_shape):
            if dim_ordering == 'th':
                return (input_shape[0],
                        input_shape[1],
                        size[0] * input_shape[2] if  input_shape[2] else None,
                        size[1] * input_shape[3] if  input_shape[3] else None)
            elif dim_ordering == 'tf':
                return (input_shape[0],
                        size[0] * input_shape[1] if  input_shape[1] else None,
                        size[1] * input_shape[2] if  input_shape[2] else None,
                        input_shape[3])
            else:
                raise Exception('Invalid dim_ordering: ' + dim_ordering)
        output = T.resize_images(inputs, size[0], size[1],
                               dim_ordering = dim_ordering)
        output._keras_shape = get_output_shape_for(input_shape)
        return output
    return func   
def Resize2D(destin_shape,dim_ordering='th',mask=None):
    def func(inputs):
        tensor_shape =  list(T.shape(inputs))
        input_shape = inputs._keras_shape
        def get_output_shape_for(input_shape):
            tmp_output_shape = destin_shape
            if dim_ordering == 'th':
                destsize = tmp_output_shape[2:4]
            elif dim_ordering == 'tf':
                destsize = tmp_output_shape[1:3]
            width =  destsize[0]
            height = destsize[1]
            if dim_ordering == 'th':
                return (input_shape[0],
                        input_shape[1],
                        width,
                        height)
            elif dim_ordering == 'tf':
                return (input_shape[0],
                        width,
                        height,
                        input_shape[3])
            else:
                raise Exception('Invalid dim_ordering: ' + dim_ordering)

        tmp_output_shape = list(destin_shape)
        if dim_ordering == 'th':
            destsize = tmp_output_shape[2:4]
        elif dim_ordering == 'tf':
            destsize = tmp_output_shape[1:3]

        if dim_ordering == 'th':
            destsize = list(destsize)
            row_residual = (destsize[0] - tensor_shape[2])
            col_residual = (destsize[1] - tensor_shape[3])
        elif dim_ordering == 'tf':
            row_residual = (destsize[0] - tensor_shape[1])
            col_residual = (destsize[1] - tensor_shape[2])

        padding = [row_residual//2,col_residual//2,  row_residual - row_residual//2,  col_residual - col_residual//2]
        cropping = [(-row_residual)//2, (-col_residual)//2,  -(row_residual + (-row_residual)//2),  -(col_residual + (-col_residual)//2)]
        #result = K.ifelse(K.gt(row_residual, 0), K.spatial_2d_padding_4specify(X, padding = padding), K.spatial_2d_cropping_4specify(X, cropping = cropping))
        result = T.spatial_2d_padding_4specify(inputs, padding = padding)
        #result = theano.printing.Print('Finish calculating resize')(result + 1)
        result._keras_shape = get_output_shape_for(input_shape)
        return result

    return func