# -*- coding: utf-8 -*-
#from numpy import *
#from theano.tensor import *
import os

if 'debug_mode' not in os.environ:
    os.environ['debug_mode'] = 'False'

if os.environ['debug_mode'] == 'True':
    print('Using numpy backend')
    from backend.numpy_backend import *
else:
    print('Using theano backend')
    from backend.theano_backend import *

from backend.numpy_backend import npwrapper
