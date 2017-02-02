import ..backend.export as T
import numpy as np 
from collections import OrderedDict
import warnings
import re
import copy

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for _, vv in tparams.iteritems()]

# initialize Theano shared variables according to the initial parameters
def update_tparams(tparams=None,params=None):
    if tparams is None:
        tparams = OrderedDict()
    for kk, vv in params.iteritems():
        if kk in tparams:
            t_trainable =  getattr(tparams[kk],'trainable', True)
        else:
            t_trainable =  getattr(vv,'trainable', True)
        #print kk
        tparams[kk] = T.variable(vv, name=kk)
        tparams[kk].trainable = t_trainable
    return tparams

def get_params(tparams): 
    new_params = OrderedDict()
    for kk, vv in tparams.iteritems():
        new_params[kk] = T.get_value(vv)
    return new_params

def init_tparams(params):
    return update_tparams(tparams = None, params=params)

def updateModuleInfo(options=None, tparams=None, prefix=None, module_identifier=None):
    ''' 
    update the current Module weights based on the tparams.
    '''
    thisModule = options[module_identifier]
    thisModule.build = True 
    sub_Params = get_subdict_prefix(tparams, prefix)
    for _, v in sub_Params.iteritems():
        if getattr(v, 'trainable', True) is True:
            thisModule.trainable_weights.append(v)
        else:
            thisModule.non_trainable_weights.append(v)

def update_or_init_params(tparams=None, params=None, tmp_params=None):
    ''' 
    all of the above three params should be given as OrderedDict.
    '''
    # we need to make sure tmp_params.trainable is passed through.
    tmp_params_keys = tmp_params.keys()
    if bool(params):
        if set(tmp_params_keys).issubset(params.keys()):
            # if params is given, we update tparams from params
            for k,v in tmp_params.iteritems():
                trainable = v.trainable
                tmp_params[k] = params[k]
                tmp_params[k].trainable = trainable
            tparams = update_tparams(tparams, tmp_params) 
        else:
            # in this case, we may need to directly update from params to tparams?
            tparams = update_tparams(tparams, tmp_params)
            for k,v in tmp_params.iteritems():
                params[k] = v
    else:
        # if params does not has all the keys, we need to initliaze it    
        if set(tmp_params_keys).issubset(tparams.keys()):
            # we don't need to do anything, just use tparams
            pass
        else:
            # if tparams does not has all the keys, it means we need to initlize params and then tparams
            tparams = update_tparams(tparams, tmp_params)
            for k,v in tmp_params.iteritems():
                params[k] = v
    return tparams, params

def get_layer_identifier(prefix):
    return 'layer_' + prefix

def get_module_identifier(prefix):
    return 'module_' + prefix
        
class obj(object):
    pass

def init_LayerInfo(options, name):
    if not name  in options:
        thisModule = obj()
        thisModule.build = False
        thisModule.trainable_weights = []
        thisModule.non_trainable_weights = []
        thisModule.regularizers = []
        thisModule.constraints = []
        thisModule.fathers = []
        options[name] = thisModule
    return options

def init_ModuleInfo(options, name):
    if not name  in options:
        thisModule = obj()
        thisModule.build = False
        thisModule.trainable_weights = []
        thisModule.non_trainable_weights = []
        thisModule.regularizers = []
        thisModule.constraints = []
        thisModule.containers = []
        thisModule.fathers = []
        options[name] = thisModule
    return options

def build_or_not(module_identifier, options):
    '''To decide do you wanna build this module'''
    
    if not module_identifier in options:
        return True
    else:
        thisModule = options[module_identifier]
        return not thisModule.build
        
def update_father_module(options,belonging_Module, module_identifier):
    '''
    we need to update the father module which contains this layer based on the information
    in this layer
    '''
    if belonging_Module is not None:
        # means we need to update the father module info mation here
        if not belonging_Module in options:
            warnings.warn('father module: {m} not initialized before calling: {f}'.
                         format(m=belonging_Module,f=module_identifier))
            init_ModuleInfo(options, belonging_Module)
        thisInfo = options[module_identifier]
        thisInfo.fathers.append(belonging_Module)
        
        belongingModuleInfo = options[belonging_Module]
        belongingModuleInfo.trainable_weights += thisInfo.trainable_weights
        belongingModuleInfo.non_trainable_weights += thisInfo.non_trainable_weights
        belongingModuleInfo.regularizers += thisInfo.regularizers
        belongingModuleInfo.constraints  += thisInfo.constraints
        belongingModuleInfo.containers.append(module_identifier)
        return options 

def get_subdict_prefix(tparams, prefixlist=None):
    ''' 
    this function is used to substract the param dictioary based on the list of prefixlist.
    '''
    if not isinstance(prefixlist, list):
        prefixlist = [prefixlist]
    prefixlist = tuple(prefixlist)
    subParams = OrderedDict()

    for k in tparams.keys():
        if k.startswith(prefixlist):
            subParams[k] = tparams[k]
    return subParams


