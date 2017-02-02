
import numpy as np
import ..backend.export as T
from   ..backend.export import npwrapper
from   .frame import *
from ..utils.utils_func import slice_tensor
#import theano
from ..utils import activations, initializations, regularizers

# -------------------------LSTM Layer------------------------------
# This function implements the lstm fprop
def init_lstm(options, params, prefix='lstm', nin=None, dim=None,
              init='norm_weight', inner_init='ortho_weight',
              forget_bias_init='one', trainable=True, **kwargs):
    init = initializations.get(init)
    inner_init = initializations.get(inner_init)
    forget_bias_init = initializations.get(forget_bias_init)

    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    """
     Stack the weight matricies for all the gates
     for much cleaner code and slightly faster dot-prods
    """
    # input weights
    W = np.concatenate([init((nin, dim), symbolic=False),
                        init((nin, dim), symbolic=False),
                        init((nin, dim), symbolic=False),
                        init((nin, dim), symbolic=False)], axis=1)

    params[get_name(prefix, 'W')] = npwrapper(W, trainable=trainable)
    # for the previous hidden activation
    U = np.concatenate([inner_init((dim, dim), symbolic=False),
                        inner_init((dim, dim), symbolic=False),
                        inner_init((dim, dim), symbolic=False),
                        inner_init((dim, dim), symbolic=False)], axis=1)
    params[get_name(prefix, 'U')] = npwrapper(U, trainable=trainable)

    b = np.hstack((np.zeros(dim),
                   forget_bias_init((dim), symbolic=False) + 4,
                   np.zeros(dim),
                   np.zeros(dim)))
    # b  = (forget_bias_init((4 * dim,) + 4,symbolic=False)+ 4)
    params[get_name(prefix, 'b')] = npwrapper(b, trainable=trainable)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None,
               init_memory=None, init_state=None, activation='tanh',
               inner_activation='hard_sigmoid', **kwargs):
    '''
    tparams: contains the ordredDict of symbolic parameters.
    state_below: timestep * batchsize * input_dim
    options: model configuration
    '''

    def get_dropout(shapelist=[None, None], dropoutrate=0):
        # if self.seed is None:
        if dropoutrate is not None and dropoutrate != False:
            retain_prob = 1 - dropoutrate

            W1 = T.binomial(shape=shapelist[0], p=retain_prob, dtype=T.floatX) / retain_prob
            W2 = T.binomial(shape=shapelist[1], p=retain_prob, dtype=T.floatX) / retain_prob
            return [W1, W2]
        else:
            return [None, None]

    activation = activations.get(activation)
    inner_activation = activations.get(inner_activation)

    nsteps = state_below.shape[0]
    dim = tparams[get_name(prefix, 'U')].shape[0]

    has_keras_shape = False
    if hasattr(state_below, '_keras_shape'):
        has_keras_shape = True
        state_keras_shape = state_below._keras_shape

    # if we are dealing with a mini-batch
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        # initial/previous state
        if init_state is None:
            init_state = T.alloc(0., (n_samples, dim), broadcastable=True)
        # initial/previous memory
        if init_memory is None:
            init_memory = T.alloc(0., (n_samples, dim), broadcastable=True)
    else:
        raise Exception('Only support 3D input')

    w_shape = (n_samples, state_below.shape[2])
    u_shape = (n_samples, tparams[get_name(prefix, 'U')].shape[0])
    dropoutmatrix = get_dropout(shapelist=[w_shape, u_shape],
                                dropoutrate=options['lstm_dropout'])

    # if we have no mask, we assume all the inputs are valid
    if mask == None:
        mask = T.alloc(1., (state_below.shape[0], 1), broadcastable=True)

    # use the slice to calculate all the different gates
    def slice_tensor(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        elif _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        return _x[n * dim:(n + 1) * dim]

    # one time step of the lstm
    def _step(m_, x_, h_, c_):
        if dropoutmatrix[1] is not None:
            drop_h_ = h_ * dropoutmatrix[1]
        else:
            drop_h_ = h_
        preact = T.dot(T.in_train_phase(drop_h_, h_), tparams[get_name(prefix, 'U')])
        preact += x_

        i = inner_activation(slice_tensor(preact, 0, dim))
        f = inner_activation(slice_tensor(preact, 1, dim))
        o = inner_activation(slice_tensor(preact, 2, dim))
        c = activation(slice_tensor(preact, 3, dim))

        c = f * c_ + i * c
        # c = m_[:,None] * c + (1. - m_)[:,None] * c_ # add by ypxie
        c = T.switch(m_, c, c_)  # add by ypxie

        h = o * activation(c)
        h = T.switch(m_, h, h_)  # add by ypxie
        # h = m_[:,None] * h + (1. - m_)[:,None] * h_  #add by ypxie

        return h, c, i, f, o, preact

    if dropoutmatrix[0] is not None:
        drop_state_below = state_below * dropoutmatrix[0]
    else:
        drop_state_below = state_below

    state_below = T.in_train_phase(drop_state_below, state_below)
    state_below = T.dot(state_below, tparams[get_name(prefix, 'W')]) + tparams[get_name(prefix, 'b')]

    rval, updates = T.scan(_step,
                           sequences=[mask, state_below],
                           outputs_info=[init_state, init_memory, None, None, None, None],
                           name=get_name(prefix, '_layers'),
                           n_steps=nsteps, profile=False)

    if has_keras_shape:
        hid_dim = T.get_value(tparams[get_name(prefix, 'U')]).shape[0]
        out_keras_shape = tuple(state_keras_shape[0:-1]) + (hid_dim,)
        rval[0] = T.add_keras_shape(rval[0], keras_shape=out_keras_shape)
    return rval

def LSTM(options, prefix='lstm', nin=None, dim=None,
         init='norm_weight', inner_init='ortho_weight', forget_bias_init='one',
         inner_activation='sigmoid', trainable=True, mask=None,
         init_memory=None, init_state=None, activation='tanh',
         nner_activation='hard_sigmoid', belonging_Module=None, **kwargs):
    '''
    params > tparams > empty
    if params covers all the weights_keys. use params to update tparams.
    '''
    tmp_params = OrderedDict()
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if 'belonging_Module' in options else None

    def func(x, tparams, options, params=None):
        tmp_params = OrderedDict()
        module_identifier = get_layer_identifier(prefix)

        if build_or_not(module_identifier, options):
            init_LayerInfo(options, name=module_identifier)
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
            nin = input_shape[-1]
        tmp_params = init_lstm(options, tmp_params, prefix=prefix, nin=nin, dim=dim, init=init, inner_init=inner_init,
                               forget_bias_init=forget_bias_init, trainable=trainable, **kwargs)
        update_or_init_params(tparams, params, tmp_params=tmp_params)

        output = lstm_layer(tparams, x, options, prefix=prefix, mask=mask, init_memory=init_memory,
                            init_state=init_state, activation=activation, inner_activation=inner_activation,
                            **kwargs)
        if build_or_not(module_identifier, options):
            updateModuleInfo(options, tparams, prefix, module_identifier)
            update_father_module(options, belonging_Module, module_identifier)
        return output
    return func

# Conditional LSTM layer with Attention
def init_dynamic_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None,
                           init='norm_weight', inner_init='ortho_weight', forget_bias_init='one',
                           ctx_dim_list=None, proj_ctx_dim=None, trainable=True, **kwargs):
    init = initializations.get(init)
    inner_init = initializations.get(inner_init)
    forget_bias_init = initializations.get(forget_bias_init)

    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if ctx_dim_list is None:
        ctx_dim_list = [options['dim'] for _ in range(options['num_attention'])]
    if proj_ctx_dim is None:
        proj_ctx_dim = options['dim']

    W = np.concatenate([init((nin, dim), symbolic=False),
                        init((nin, dim), symbolic=False),
                        init((nin, dim), symbolic=False),
                        init((nin, dim), symbolic=False)], axis=1)

    params[get_name(prefix, 'W')] = npwrapper(W, trainable=trainable)
    # for the previous hidden activation
    U = np.concatenate([inner_init((dim, dim), symbolic=False),
                        inner_init((dim, dim), symbolic=False),
                        inner_init((dim, dim), symbolic=False),
                        inner_init((dim, dim), symbolic=False)], axis=1)
    params[get_name(prefix, 'U')] = npwrapper(U, trainable=trainable)
    b = np.hstack((np.zeros(dim),
                   forget_bias_init((dim), symbolic=False),
                   np.zeros(dim),
                   np.zeros(dim)))
    # bias to LSTM
    params[get_name(prefix, 'b')] = npwrapper(b, trainable=trainable)

    # context to LSTM hidden
    for idx, ctx_dim in enumerate(ctx_dim_list):

        if options['project_context']:
            mutable_ctx_dim = ctx_dim
            # attention: context -> project context
            if options['n_layers_att'] > 0:
                for lidx in xrange(0, options['n_layers_att']):
                    W_c2pc = init((mutable_ctx_dim, proj_ctx_dim), symbolic=False)
                    params[get_name(prefix, 'W_c2pc_{att_num}_cidx_{cidx}'.format(att_num=lidx, cidx=idx))] = npwrapper(
                        W_c2pc, trainable=trainable)
                    # attention: hidden bias
                    b_c2pc = initializations.get('zero')((proj_ctx_dim,), symbolic=False).astype(T.floatX)
                    params[get_name(prefix, 'b_c2pc_{att_num}_cidx_{cidx}'.format(att_num=lidx, cidx=idx))] = npwrapper(
                        b_c2pc, trainable=trainable)
                    mutable_ctx_dim = proj_ctx_dim
                    
        else:
            proj_ctx_dim = ctx_dim

        Wc2h = init((ctx_dim, dim * 4), symbolic=False)
        params[get_name(prefix, 'Wc2h_cidx_%d' % idx)] = npwrapper(Wc2h, trainable=trainable)

        if options['attn_type'] == 'dynamic':
            # get_w  parameters for reading operation
            if options['addressing'] == 'softmax':
                params[get_name(prefix, 'W_k_read_cidx_%d' % idx)] = npwrapper(
                    inner_init((dim, proj_ctx_dim), symbolic=False), trainable=trainable)
                params[get_name(prefix, 'b_k_read_cidx_%d' % idx)] = npwrapper(
                    initializations.get('zero')((proj_ctx_dim), symbolic=False), trainable=trainable)
                params[get_name(prefix, 'W_address_cidx_%d' % idx)] = npwrapper(
                    inner_init((proj_ctx_dim, proj_ctx_dim), symbolic=False), trainable=trainable)

            elif options['addressing'] == 'ntm':
                params[get_name(prefix, 'W_k_read_cidx_%d' % idx)] = npwrapper(
                    inner_init((dim, proj_ctx_dim), symbolic=False), trainable=trainable)
                params[get_name(prefix, 'b_k_read_cidx_%d' % idx)] = npwrapper(
                    initializations.get('zero')((proj_ctx_dim), symbolic=False), trainable=trainable)
                params[get_name(prefix, 'W_c_read_cidx_%d' % idx)] = npwrapper(inner_init((dim, 3), symbolic=False),
                                                                               trainable=trainable)
                params[get_name(prefix, 'b_c_read_cidx_%d' % idx)] = npwrapper(
                    initializations.get('zero')((3,), symbolic=False), trainable=trainable)
                params[get_name(prefix, 'W_s_read_cidx_%d' % idx)] = npwrapper(
                    inner_init((dim, options['shift_range']), symbolic=False), trainable=trainable)
                params[get_name(prefix, 'b_s_read_cidx_%d' % idx)] = npwrapper(
                    initializations.get('zero')((options['shift_range'],), symbolic=False),
                    trainable=trainable)
        else:
            # attention:
            U_att = init((proj_ctx_dim, 1), symbolic=False)
            params[get_name(prefix, 'U_att_cidx_%d' % idx)] = npwrapper(U_att, trainable=trainable)
            c_att = initializations.get('zero')((1,), symbolic=False)
            params[get_name(prefix, 'c_tt_cidx_%d' % idx)] = npwrapper(c_att, trainable=trainable)
            # attention: hidden-->project_ctx
            W_h2pc = init((dim,proj_ctx_dim),symbolic=False)
            params[get_name(prefix,'W_h2pc_cidx_%d'%idx)] =  npwrapper(W_h2pc,trainable=trainable)

        if options['selector']:
            # attention: selector
            W_sel = init((dim, 1), symbolic=False)
            params[get_name(prefix, 'W_sel_cidx_%d' % idx)] = npwrapper(W_sel, trainable=trainable)
            b_sel = np.float32(0.).astype(T.floatX)
            params[get_name(prefix, 'b_sel_cidx_%d' % idx)] = npwrapper(b_sel, trainable=trainable)

    return params


def dynamic_lstm_cond_layer(tparams, state_below, options, prefix='dlstm', mask=None, context_list=[None],
                            last_step=False, init_memory=None, init_state=None, init_alpha_sampling_list=None,
                            rng=None, sampling=True, argmax=False, activation='tanh',
                            inner_activation='sigmoid', **kwargs):
    '''
    Parameters
    ----------
      tparams: contains the ordredDict of symbolic parameters.
      state_below: timestep * batchsize * input_dim
      context_list : list of three [nsample * annotation * dim], currently it is of three hierarchical memory
      options: model configuration
    Returns
    -------
    '''

    def get_dropout(shapelist=[None], dropoutrate=0):
        # if self.seed is None:
        if dropoutrate:
            retain_prob = 1 - dropoutrate
            # retain_prob_U = 1- dropoutrate[0]
            W1 = T.binomial(shape=shapelist[0], p=retain_prob, dtype=T.floatX) / retain_prob
            # W2 = T.binomial(shape= shapelist[0], p = retain_prob, dtype = T.floatX)/retain_prob
            # W3 = T.binomial(shape= shapelist[2], p = retain_prob, dtype = T.floatX)/retain_prob
            return [W1]
        else:
            return [None]

    def check_ndim(cks, ndim):
        if type(cks) == list:
            return all([T.ndim(ck) == ndim for ck in cks])
        else:
            return T.ndim(cks) == ndim

    def wrapper_rval(rval, num_attention):
        wrapped_rval = [None] * 5
        interval = num_attention
        wrapped_rval[0:2] = rval[0:2]
        wrapped_rval[2] = [rval[2 + idx] for idx in range(interval)]  # list of alpha
        wrapped_rval[3] = [rval[2 + interval + idx] for idx in range(interval)]  # list of alpha samples
        wrapped_rval[4] = [rval[2 + interval * 2 + idx] for idx in range(interval)]  # list of att_ctx
        #wrapped_rval[5] = [rval[2 + interval * 3 + idx] for idx in range(interval)]  # list of sel_
        return wrapped_rval

    activation = activations.get(activation)
    inner_activation = activations.get(inner_activation)
    k_activ = activations.get(options['k_activ']) if 'k_activ' in options else T.tanh

    if check_ndim(state_below, 2):
        state_below = T.expand_dims(state_below, dim=0, broadcastable = True)
        print 'state_below has only dimensional 2, we pad the first dimension'
        assert check_ndim(state_below, 3), 'state_below dimension is not 3'

    state_keras_shape = getattr(state_below, '_keras_shape', (None,None,None))

    for idx, context in enumerate(context_list):
        # context should be in batchsize * num_positions * dim
        assert (context is not None), 'context can not be nan'
        # assert check_ndim(context,3), 'context dimension is not 3'
        if T.ndim(context) == 2:
            context = T.expand_dims(context, dim=0)
            real_ctx_dim = context._keras_shape[-1]
            destin_shape = (state_below.shape[1], context.shape[-2], context.shape[-1])
            context = context + T.alloc(0., destin_shape)
            context._keras_shape = (None,None, real_ctx_dim)
        assert not T.isnan(context), 'Context must be provided'
        context_list[idx] = context


    n_samples = state_below.shape[1]
    nsteps = state_below.shape[0]

    if mask is None:
        mask = T.alloc(1., (nsteps, n_samples), broadcastable=True)
    else:
        assert check_ndim(mask, 2), 'mask dimension is not 2'
    dim = tparams[get_name(prefix, 'U')].shape[1]
    # initial/previous state
    if init_state is None:
        init_state = T.alloc(0., (n_samples, dim), broadcastable=True)
    else:
        assert check_ndim(init_state, 2), 'init_state dimension is not 2'
    # initial/previous memory
    if init_memory is None:
        init_memory = T.alloc(0., (n_samples, dim), broadcastable=True)
    else:
        assert check_ndim(init_memory, 2), 'init_memory dimension is not 2'

    # w_shape   = (n_samples, state_below.shape[-1])
    # att_shape = (1,batchsize, tparams[get_name(prefix,'W_h2pc')].shape[0])
    u_shape = (n_samples, 3 * tparams[get_name(prefix, 'U')].shape[0])
    # ctx_shape = (n_samples, tparams[get_name(prefix, 'Wc')].shape[0])

    # projected x
    # state_below is timesteps*num samples by d in training 
    # this is n * d during sampling
    dropoutmatrix = get_dropout(shapelist=[u_shape], dropoutrate=options['lstm_dropout'])
    # drop_state_below   =   state_below *dropoutmatrix[0] if dropoutmatrix[0] is not None else state_below
    # tate_below = T.in_train_phase(drop_state_below, state_below)

    #state_below = theano.printing.Print("this is state_below before W")(state_below + 1e-9)
    state_below = T.dot(state_below, tparams[get_name(prefix, 'W')]) + tparams[get_name(prefix, 'b')]
    #state_below = theano.printing.Print("this is state_below")(state_below + 1e-9)

    # infer lstm dimension
    pctx_collection = []
    ctx_keras_shape_list = []

    dim = tparams[get_name(prefix, 'U')].shape[0]
    for idx, context in enumerate(context_list):
        if options['project_context']:
            # projected context
            pctx_ = context
            for lidx in xrange(0, options['n_layers_att']):
                pctx_ = T.dot(pctx_,tparams[get_name(prefix, 'W_c2pc_{att_num}_cidx_{cidx}'.format(att_num=lidx, cidx=idx))]) \
                        + tparams[get_name(prefix, 'b_c2pc_{att_num}_cidx_{cidx}'.format(att_num=lidx, cidx=idx))]
                # note to self: this used to be options['n_layers_att'] - 1, so no extra non-linearity if n_layers_att < 3
                # if lidx < options['n_layers_att']:
                if lidx < options['n_layers_att'] - 1:
                    pctx_ = activation(pctx_)
                kshape = (None, None, options['proj_ctx_dim'])
                T.add_keras_shape(pctx_, keras_shape=kshape)
        else:
            pctx_ = context
        pctx_collection.append(pctx_)

    # temperature for softmax
    temperature = options.get("temperature", 1)
    temperature_c = T.shared(np.float32(temperature), name='temperature_c')
    # additional parameters for stochastic hard attention
    if options['attn_type'] == 'stochastic' or options['hard_sampling'] == True:
        # temperature for softmax
        # temperature = options.get("temperature", 1)
        # [see (Section 4.1): Stochastic "Hard" Attention]
        semi_sampling_p = options.get("semi_sampling_p", 0.5)
        h_sampling_mask = T.binomial((1,), p=semi_sampling_p, n=1, dtype=T.floatX, rng=rng).sum()

    def _step(m_, x_, h_, c_, wr_tm1, wr_tm2, wr_tm3, pctx_1, pctx_2, pctx_3):
        """ Each variable is one time slice of the LSTM
        Only use it if you use wr_tm1, otherwise use a wrapper that does not have wr_tm1
        m_ - (mask), x_- (previous word),  
        h_- (hidden state), c_- (lstm memory),
        wr_tm1(as_)- (sample from alpha dist),
        pctx_1, pctx_2, pctx_3 are just non-sequences represented the non-changable context feature.

        it returns:
        rval = [h, c, alpha, alpha_sample, att_ctx]
        if options['selector']:
            rval += [sel_]
        return rval
        """
        num_attention = options['num_attention']
        # alpha_collect = [a_1,a_2,a_3][0:num_attention]
        wr_tm_collect = [wr_tm1, wr_tm2, wr_tm3][0:num_attention]
        pctx_collect = [pctx_1, pctx_2, pctx_3][0:num_attention]

        def get_ntm_alpha(options, h_, pctx_collect, wr_tm_collect, temperature_c):
            alpha_list = []
            for idx, pctx_ in enumerate(pctx_collect):
                wr_tm1 = wr_tm_collect[idx]
                W_k_read = tparams[get_name(prefix, 'W_k_read_cidx_%d' % idx)]
                b_k_read = tparams[get_name(prefix, 'b_k_read_cidx_%d' % idx)]
                W_c_read = tparams[get_name(prefix, 'W_c_read_cidx_%d' % idx)]
                b_c_read = tparams[get_name(prefix, 'b_c_read_cidx_%d' % idx)]
                W_s_read = tparams[get_name(prefix, 'W_s_read_cidx_%d' % idx)]
                b_s_read = tparams[get_name(prefix, 'b_s_read_cidx_%d' % idx)]

                #h_ = theano.printing.Print("this is h_%d: "%idx)(h_ + 1e-9)

                k_read, beta_read, g_read, gamma_read, s_read = get_controller_output(
                    h_, W_k_read, b_k_read, W_c_read, b_c_read,
                    W_s_read, b_s_read, k_activ=k_activ)

                #k_read = theano.printing.Print("this is k_read_%d: "%idx)(k_read + 1e-9)
                #beta_read = theano.printing.Print("this is beta_read_%d: "%idx)(beta_read + 1e-9)

                C = circulant(pctx_.shape[1], options['shift_range'])
                wc_read = get_content_w(beta_read, k_read, pctx_)


                #g_read = theano.printing.Print("this is g_read_%d: "%idx)(g_read + 1e-9)
                #wc_read = theano.printing.Print("this is wc_read_%d: "%idx)(wc_read + 1e-9)
                #gamma_read = theano.printing.Print("this is gamma_read_%d: "%idx)(gamma_read + 1e-9)
                #s_read = theano.printing.Print("this is s_read_%d: "%idx)(s_read + 1e-9)


                alpha_pre = wc_read
                alpha_shp = wc_read.shape
                alpha = get_location_w(g_read, s_read, C, gamma_read,
                                       wc_read, wr_tm1)
                alpha_list.append(alpha)
            return alpha_list

        def get_softmax_alpha(options, h_, pctx_collect, temperature_c):
            alpha_list = []
            for idx, pctx_ in enumerate(pctx_collect):
                W_k_read = tparams[get_name(prefix, 'W_k_read_cidx_%d' % idx)]
                b_k_read = tparams[get_name(prefix, 'b_k_read_cidx_%d' % idx)]
                W_address = tparams[get_name(prefix, 'W_address_cidx_%d' % idx)]

                k = T.tanh(T.dot(h_, W_k_read) + b_k_read)  # + 1e-6
                score = (T.dot(pctx_, W_address) * k[:, None, :]).sum(axis=-1)  # N * location
                alpha_pre = score
                alpha_shp = alpha_pre.shape
                alpha = T.softmax(temperature_c * score)
                alpha_list.append(alpha)
            return alpha_list

        def original_alpha(options, h_, pctx_collect, temperature_c):
            alpha_list = []
            for idx, pctx_ in enumerate(pctx_collect):
                pstate_ = T.dot(h_, tparams[get_name(prefix,'W_h2pc_cidx_%d'%idx)])
                pctx_ = pctx_ + pstate_[:,None,:]
                pctx_ = activation(pctx_)
                alpha = T.dot(pctx_, tparams[get_name(prefix, 'U_att_cidx_%d' % idx)]) + tparams[
                    get_name(prefix, 'c_tt_cidx_%d' % idx)]
                # print tparams[get_name(prefix, 'c_tt_cidx_%d'%idx)].broadcastable
                alpha_pre = alpha
                alpha_shp = alpha.shape
                alpha = T.softmax(temperature_c * alpha.reshape([alpha_shp[0], alpha_shp[1]]))  # softmax
                alpha_list.append(alpha)
            return alpha_list

        if options['attn_type'] == 'dynamic':
            # get controller output
            if options['addressing'] == 'ntm':
                alpha_list = get_ntm_alpha(options, h_, pctx_collect, wr_tm_collect, temperature_c)
            elif options['addressing'] == 'softmax':
                alpha_list = get_softmax_alpha(options, h_, pctx_collect, temperature_c)
            elif options['addressing'] == 'cosine':
                pass
        else:
            alpha_list = original_alpha(options, h_, pctx_collect, temperature_c)
        alpha_sample_list = []
        att_ctx_list = []
        
        for this_context, alpha in zip(context_list, alpha_list):

            if options['hard_sampling'] == True or options['attn_type'] == 'stochastic':
                if sampling:
                    alpha_sample = h_sampling_mask * T.multinomial(pvals=alpha, dtype=T.floatX, rng=rng) \
                                   + (1. - h_sampling_mask) * alpha
                else:
                    if argmax:
                        alpha_sample = T.cast(T.eq(T.arange(alpha.shape[1])[None, :],
                                                   T.argmax(alpha, axis=1, keepdims=True)), T.floatX)
                    else:
                        alpha_sample = alpha
                att_ctx = (this_context * alpha_sample[:, :, None]).sum(axis=1)  # current context
            else:
                att_ctx = (this_context * alpha[:, :, None]).sum(axis=1)  # current context
                alpha_sample = alpha  # you can return something else reasonable here to debug
            alpha_sample_list.append(alpha_sample)
            att_ctx_list.append(att_ctx)

        sel_list = []
        if options['selector']:
            for idx, att_ctx in enumerate(att_ctx_list):
                sel_ = T.sigmoid(T.dot(h_, tparams[get_name(prefix, 'W_sel_cidx_%d' % idx)]) + tparams[
                    get_name(prefix, 'b_sel_cidx_%d' % idx)])
                sel_ = sel_.reshape([sel_.shape[0]])
                att_ctx = sel_[:, None] * att_ctx
                sel_list.append(sel_)
                att_ctx_list[idx] = att_ctx
        # we need to concatenate the feature to get a whole combined feature
        preact = T.dot(h_, tparams[get_name(prefix, 'U')])
        preact += x_
        # preact += T.dot(T.in_train_phase(drop_ctx_, att_ctx), tparams[get_name(prefix, 'Wc')])
        for idx, att_ctx in enumerate(att_ctx_list):
            preact += T.dot(att_ctx, tparams[get_name(prefix, 'Wc2h_cidx_%d' % idx)])
        # applied bayesian LSTM
        cut_preact = slice_tensor(preact, 0, 3 * dim)
        drop_cut_preact = cut_preact * dropoutmatrix[0] if dropoutmatrix[0] is not None else cut_preact
        cut_preact = T.in_train_phase(drop_cut_preact, cut_preact)

        i = inner_activation(slice_tensor(cut_preact, 0, dim))
        f = inner_activation(slice_tensor(cut_preact, 1, dim))
        o = inner_activation(slice_tensor(cut_preact, 2, dim))
        c = activation(slice_tensor(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * activation(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        rval = [h, c] + alpha_list + alpha_sample_list + att_ctx_list

        #if options['selector']:
        #    rval.extend(sel_list)
        # rval += [pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
        return rval

    if options['num_attention'] == 2:
        def step_func(m_, x_, h_, c_, wr_tm1, wr_tm2, pctx_1, pctx_2):
            return _step(m_, x_, h_, c_, wr_tm1, wr_tm2, None, pctx_1, pctx_2, None)
    if options['num_attention'] == 1:
        def step_func(m_, x_, h_, c_, wr_tm1, pctx_1):
            return _step(m_, x_, h_, c_, wr_tm1, None, None, pctx_1, None, None)
    if options['num_attention'] == 3:
        step_func = _step
    seqs = [mask, state_below]
    outputs_info = [init_state,  # h
                    init_memory  # c
                    ]
    outputs_info += [None for pctx_ in pctx_collection]  # alpha_
    if init_alpha_sampling_list is None:
        outputs_info += [T.alloc(0., (n_samples, pctx_.shape[1]), broadcastable=True) for pctx_ in
                         pctx_collection]  # as_
    else:
        outputs_info += init_alpha_sampling_list

    outputs_info += [None for context in context_list]  # ct_
    #if options['selector']:
    #    outputs_info += [None for _ in range(len(pctx_collection))]
    rval, updates = T.scan(step_func,
                           sequences=seqs,
                           outputs_info=outputs_info,
                           non_sequences=pctx_collection,
                           name=get_name(prefix, '_layers'),
                           n_steps=nsteps, profile=False)
    # add the additional _keras_shape_info
    hid_dim = T.get_value(tparams[get_name(prefix, 'U')]).shape[0]
    out_keras_shape = tuple(state_keras_shape[0:-1]) + (hid_dim,)
    rval[0] = T.add_keras_shape(rval[0], keras_shape=out_keras_shape)

    start_ind = 2 + options['num_attention'] * 2
    for idx, ctx in enumerate(context_list):
        if hasattr(ctx, '_keras_shape'):
            ctx_keras_shape = ctx._keras_shape
            out_ctx_keras_shape = tuple(state_keras_shape[0:-1]) + (ctx_keras_shape[-1],)
            rval[start_ind + idx] = T.add_keras_shape(rval[start_ind + idx], keras_shape=out_ctx_keras_shape)

    if last_step == True:
        for idx, rv in enumerate(rval):
            rval[idx] = rv[-1]
    wrapped_rval = wrapper_rval(rval, options['num_attention'])
    return wrapped_rval, updates

def cond_LSTM(options, prefix='lstm_cond', nin=None, dim=None, init='norm_weight', inner_init='ortho_weight',
              forget_bias_init='one', ctx_dim_list=None, proj_ctx_dim=None, trainable=True, mask=None,
              context_list=[None], one_step=False, init_memory=None, init_state=None, rng=None, sampling=True,
              init_alpha=None, argmax=False, activation='tanh', inner_activation='sigmoid', belonging_Module=None,
              **kwargs):
    '''
    params > tparams > empty
    if params covers all the weights_keys. use params to update tparams.
    '''
    if not belonging_Module:
        belonging_Module = options['belonging_Module'] if 'belonging_Module' in options else None
    tmpDict = {'nin': nin, 'ctx_dim_list': ctx_dim_list}

    def func(x, tparams, options, params=None):
        module_identifier = get_layer_identifier(prefix)
        tmp_params = OrderedDict()

        if build_or_not(module_identifier, options):
            init_LayerInfo(options, name=module_identifier)

        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
            tmpDict['nin'] = input_shape[-1]
        if tmpDict['ctx_dim_list'] is None:
            tmpDict['ctx_dim_list'] = [None for _ in context_list]
        else:
            assert len(tmpDict['ctx_dim_list']) != len(
                context_list), "contex_dim_list does not have same length with ctx_dim_list"
        ctx_dim_list = tmpDict['ctx_dim_list']

        for idx, context in enumerate(context_list):
            if hasattr(context, '_keras_shape'):
                contex_shape = context._keras_shape
                ctx_dim_list[idx] = contex_shape[-1]

        tmp_params = init_dynamic_lstm_cond(options, tmp_params, prefix=prefix, nin=tmpDict['nin'], dim=dim, init=init,
                                            inner_init=inner_init, forget_bias_init=forget_bias_init,
                                            ctx_dim_list=ctx_dim_list, proj_ctx_dim=proj_ctx_dim,
                                            trainable=trainable, **kwargs)
        update_or_init_params(tparams, params, tmp_params=tmp_params)

        output = dynamic_lstm_cond_layer(tparams, x, options, prefix=prefix, mask=mask,
                                         context_list=context_list, one_step=one_step, init_memory=init_memory,
                                         init_state=init_state, rng=rng, sampling=sampling,
                                         init_alpha=init_alpha, argmax=argmax,
                                         activation=activation, inner_activation=inner_activation,
                                         **kwargs)
        if build_or_not(module_identifier, options):
            updateModuleInfo(options, tparams, prefix, module_identifier)
            update_father_module(options, belonging_Module, module_identifier)
        return output

    return func
