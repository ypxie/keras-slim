from six import string_types, iteritems
import copy
import numpy as np
import logging
import warnings
from collections import OrderedDict

def safe_new(x, tag='', dtype=None):
    """
    Internal function that constructs a new variable from x with the same
    type, but with a different name (old name + tag). This function is used
    by gradient, or the R-op to construct new variables for the inputs of
    the inner graph such that there is no interference between the original
    graph and the newly constructed graph.

    """
    if hasattr(x, 'name') and x.name is not None:
        nw_name = x.name + tag
    else:
        nw_name = None

    if dtype and x.dtype != dtype:
        casted_x = x.astype(dtype)
        nwx = x.__class__(casted_x.type, x.data, x.name)
        nwx.tag = copy(x.tag)
        return nwx
    else:
        return x.copy()

def expand_empty(tensor_var, size):
    """
    Transforms the shape of a tensor from (d1, d2 ... ) to ( d1+size, d2, ..)
    by adding uninitialized memory at the end of the tensor.

    """

    if size == 0:
        return tensor_var
    shapes = [tensor_var.shape[x] for x in xrange(tensor_var.ndim)]
    new_shape = [size + shapes[0]] + shapes[1:]
    empty = np.zeros((new_shape), dtype = tensor_var.dtype)
    empty[:shapes[0]] = tensor_var
    #empty.tag.nan_guard_mode_check = False
    return empty
    

def isNaN_or_Inf_or_None(x):
    isNone = x is None
    try:
        isNaN = np.isnan(x)
        isInf = np.isinf(x)
        isStr = isinstance(x, string_types)
    except Exception:
        isNaN = False
        isInf = False
        isStr = False
    if not isNaN and not isInf:
        try:
            isInf = np.isinf(x)
            isNaN = np.isnan(x)
        except Exception:
            isNaN = False
            isInf = False
    if isinstance(x, string_types):
        isStr = True
    else:
        isStr = False
    return isNone or isNaN or isInf or isStr
    

class until(object):
    """
    Class used to encode the different things the inner function of scan can
    (or needs) to return.

    This class has to be used when scan needs to halt when a condition is
    met, otherwise the list of outputs and dictionary can directly be return
    as a tuple. The reason is that otherwise scan has no way to distinguish
    between the condition and the list of outputs ( unless we enforce and
    order, but since this was not impose up to know it can make quite a bit
    of code to fail).

    """

    def __init__(self, condition):
        self.condition = np.asarray(condition)
        assert self.condition.ndim == 0    

def get_updates_and_outputs(ls):
    """
    This function tries to recognize the updates OrderedDict, the
    list of outputs and the stopping condition returned by the
    lambda expression and arrange them in a predefined order.

    WRITEME: what is the type of ls? how is it formatted?
            if it's not in the predefined order already, how does
            this function know how to put it in that order?

    """
    def is_outputs(elem):
        if (isinstance(elem, (list, tuple)) and
            all([isinstance(x, np.ndarray) for x in elem])):
            return True
        if isinstance(elem, np.ndarray):
            return True
        return False

    def is_updates(elem):
        if isinstance(elem, dict):
            # Make sure the updates will be applied in a deterministic order
            if (not isinstance(elem, OrderedDict) and
                len(elem) > 1):
                warnings.warn("Expected OrderedDict or OrderedUpdates, got "\
                        + str(type(elem)) + ". This can make your script non-"
                        "deterministic.")
            return True
        # Dictionaries can be given as lists of tuples
        if (isinstance(elem, (list, tuple)) and
            all([isinstance(x, (list, tuple)) and len(x) == 2
                 for x in elem])):
            return True
        return False

    def is_condition(elem):
        return isinstance(elem, until)

    def _list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        else:
            return [x]

    def _filter(x):
        """
        Ensure `x` is made only of allowed data types.

        Return True iff `x` is made only of lists, tuples, dictionaries, Theano
        variables or `theano.scan_module.until` objects.

        """
        # Is `x` a container we can iterate on?
        iter_on = None
        if isinstance(x, list) or isinstance(x, tuple):
            iter_on = x
        elif isinstance(x, dict):
            iter_on = iteritems(x)
        if iter_on is not None:
            return all(_filter(y) for y in iter_on)
        else:
            return (isinstance(x, np.ndarray) or
                    isinstance(x, until))

    if not _filter(ls):
        raise ValueError(
                'The return value of your scan lambda expression may only be '
                'made of lists, tuples, or dictionaries containing Theano '
                'variables (or `theano.scan_module.until` objects for '
                'conditions). In particular if you need to use constant '
                'values, you can use `tensor.constant` to turn them into '
                 'Theano variables.')

    if is_outputs(ls):
        return None, _list(ls), OrderedDict()
    if is_updates(ls):
        return None, [], OrderedDict(ls)
    error_msg = ('Scan cannot parse the return value of your lambda '
                 'expression, which is: %s' % (ls,))
    if not isinstance(ls, (list, tuple)):
        raise ValueError(error_msg)
    ls = list(ls)
    deprecation_msg = ('The return value of the lambda function'
                    ' has been restricted. you have to always return first the'
                    ' outputs (if any), afterwards the updates (if any) and'
                    ' at the end the conclusion')
    if len(ls) == 2:
        if is_outputs(ls[0]):
            if is_updates(ls[1]):
                return (None, _list(ls[0]), OrderedDict(ls[1]))
            elif is_condition(ls[1]):
                return (ls[1].condition, _list(ls[0]), OrderedDict())
            else:
                raise ValueError(error_msg)
        elif is_updates(ls[0]):
            if is_outputs(ls[1]):
                raise ValueError(deprecation_msg)
            elif is_condition(ls[1]):
                return (ls[1].condition, [], OrderedDict(ls[0]))
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
    elif len(ls) == 3:
        if is_outputs(ls[0]):
            if is_updates(ls[1]):
                if is_condition(ls[2]):
                    return (ls[2].condition, _list(ls[0]), OrderedDict(ls[1]))
                else:
                    raise ValueError(error_msg)
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
    else:
        raise ValueError(error_msg)
