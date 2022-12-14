#################################################################################
# Found here: https://gist.github.com/apaszke/4c8ead6f17a781d589f6655692e7f6f0
#################################################################################

import sys
import types
from collections import OrderedDict

import torch
import numpy as np
from termcolor import colored

PY2 = sys.version_info[0] == 2
_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}


### Had to add this for conv net
_new_methods = {'conv2d_forward','_forward_impl', '_check_input_dim', '_conv_forward'}


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()


# Function keeps looping and turning each module in the network to a function
def _make_functional(module, params_box, params_offset):
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    # Set dummy variable to bias_None to rename as flag if no bias
    if 'bias' in param_names and module._parameters['bias'] is None:
        param_names[-1] = 'bias_None' # Remove last name (hopefully bias) from list
    forward = type(module).forward.__func__ if PY2 else type(module).forward
    if type(module) == torch.nn.modules.container.Sequential:
        # Patch sequential model by replacing the forward method
        forward = Sequential_forward_patch
    if 'BatchNorm' in module.__class__.__name__:
        # Patch sequential model by replacing the forward method (hoping applies for all BNs need
        # to put this in tests)
        forward = bn_forward_patch

    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue   #If internal attributes skip
        setattr(self, name, attr)
    ### Had to add this for conv net (MY ADDITION)
    for name in dir(module):
        if name in _new_methods:
            if name == '_conv_forward': # Patch for pytorch 1.5.0+cu101
                setattr(self, name, types.MethodType(type(module)._conv_forward,self))
            if name == 'conv2d_forward':
                setattr(self, name, types.MethodType(type(module).conv2d_forward,self))
            if name == '_forward_impl':
                setattr(self, name, types.MethodType(type(module)._forward_impl,self))
            if name == '_check_input_dim': # Batch Norm
                # import pdb; pdb.set_trace()
                setattr(self, name, types.MethodType(type(module)._check_input_dim,self))

    child_params_offset = params_offset + num_params
    for name, child in module.named_children():
        child_params_offset, fchild = _make_functional(child, params_box, child_params_offset)
        self._modules[name] = fchild  # fchild is functional child
        setattr(self, name, fchild)
    def fmodule(*args, **kwargs):

        # Uncomment below if statement to step through (with 'n') assignment of parameters.
#         if params_box[0] is not None:
#             import pdb; pdb.set_trace()

        # If current layer has no bias, insert the corresponding None into params_box
        # with the params_offset ensuring the correct weight is applied to the right place.
        if 'bias_None' in param_names:
            params_box[0].insert(params_offset + 1, None)
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):

            # In order to deal with layers that have no bias:
            if name == 'bias_None':
                setattr(self, 'bias', None)
            else:
                setattr(self, name, param)
        # In the forward pass we receive a context object and a Tensor containing the
        # input; we must return a Tensor containing the output, and we can use the
        # context object to cache objects for use in the backward pass.

        # When running the kwargs no longer exist as they were put into params_box and therefore forward is just
        # forward(self, x), so I could comment **kwargs out
        return forward(self, *args) #, **kwargs)

    return child_params_offset, fmodule


def make_functional(module):
    params_box = [None]
    _, fmodule_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        params_box[0] = kwargs.pop('params') # if key is in the dictionary, remove it and return its value, else return default. If default is not given and key is not in the dictionary, a KeyError is raised.
        return fmodule_internal(*args, **kwargs)

    return fmodule

##### PATCH FOR nn.Sequential #####

def Sequential_forward_patch(self, input):
    # put at top of notebook nn.Sequential.forward = Sequential_forward_patch
    for label, module in self._modules.items():
        input = module(input)
    return input

##### Patch for batch norm #####
def bn_forward_patch(self, input):
    # set running var to None and running mean
    return torch.nn.functional.batch_norm(
                input, running_mean = None, running_var = None,
                weight = self.weight, bias = self.bias,
                training = self.training,
                momentum = self.momentum, eps = self.eps)

def gpu_check_delete(string, locals):
    if string in locals:
        del locals[string]
        torch.cuda.empty_cache()
