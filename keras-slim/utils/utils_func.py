
import ..backend.export as T
from ..backend.export import npwrapper
import numpy as np
from collections import OrderedDict


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in pp.iteritems():
        #if kk not in params:
        #    raise Warning('%s is not in the archive' % kk)
        if kk not in params:
            params[kk] = npwrapper(pp[kk], trainable=True)
        else:
            if hasattr(params[kk], 'trainable'):
                params[kk] = npwrapper(pp[kk], trainable=params[kk].trainable)
            else:
                params[kk] = npwrapper(pp[kk], trainable=True)
    return params

def load_keras_model(params, keras_model_path, max_layer = None):
    '''
    Load all layer weights from a HDF5 save file.
    '''
    if params is None:
        params = OrderedDict()
    import h5py
    f = h5py.File(keras_model_path, mode='r')    
    # new file format
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']] 
    if max_layer is None:
        max_layer = len(layer_names)   
    for k, name in enumerate(layer_names):
        if k == max_layer:
            break
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            for weight_name in weight_names:
                if weight_name not in params:
                    warnings.warn("params does not have key: {s} when loading keras \
                                   model".format(s=weight_name))
                params[weight_name] = npwrapper(g[weight_name], trainable=True) 
    f.close()    
    return params
            
    
def weighted_objective(fn):
    '''Transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.
    '''
    def weighted(y_true, y_pred, weights, mask=None):
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            mask = T.cast(mask, K.floatx())
            # mask should have the same shape as score_array
            score_array *= mask
            #  the loss per batch should be proportional
            #  to the number of unmasked samples.
            score_array /= T.mean(mask)

        # reduce score_array to same ndim as weight array
        ndim = T.ndim(score_array)
        weight_ndim = T.ndim(weights)
        score_array = T.mean(score_array, axis=list(range(weight_ndim, ndim)))

        # apply sample weighting
        if weights is not None:
            score_array *= weights
            score_array /= T.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return T.mean(score_array)
    return weighted    


def split_words(words):
    return re.findall(r'\w+|\S+', words)
def multi_list(value, times, share=False):
    if share:
        return [value for _ in range(times)]
    else:
        return [copy.copy(value) for _ in range(times)]

def expand_list(nested_list, plain_list=None):
    if plain_list is None:
        plain_list = []
    for lis in nested_list:
        if type(lis) != list:
            plain_list.append(lis)
        else:
            expand_list(lis, plain_list)
    return  plain_list      
# The following functions are for 
def slice_tensor(x, n, dim):
    if T.ndim(x) == 3:
        return x[:, :, n*dim:(n+1)*dim]
    elif T.ndim(x) == 2:
        return x[:, n*dim:(n+1)*dim]
    return x[n*dim:(n+1)*dim]

