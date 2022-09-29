import sys
import random
# from callee import *
import traceback

import numpy as np
import torch
import torch.nn as nn

from .utils import *
from .gradients import *


'''For HMC leapfrog'''
def params_grad(log_prob_func, p):
    p = p.detach().requires_grad_()
    log_prob = log_prob_func(p)
    
    p = collect_gradients(log_prob, p)
    
    # DEBUG
    # print(p.grad.std())
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return p.grad

def hmc_leapfrog(params, momentum, log_prob_func, steps=10, step_size=0.1, inv_mass=None, store_on_GPU = True, debug=False):
    
    """
    This is a rather large function that contains all the various integration schemes used for HMC. Broadly speaking, it takes in the parameters
    and momentum and propose a new set of parameters and momentum. This is a key part of hamiltorch as it covers multiple integration schemes.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    steps : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    store_on_GPU : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set to TRUE but may run out of memory unless set to FALSE.
    debug : int
        This is useful for checking how many iterations RMHMC takes to converge. Set to zero for no print statements.

    Returns
    -------
    ret_params : list
            List of parameters collected in the trajectory. Note that explicit RMHMC returns a copy of two lists.
        ret_momenta : list
            List of momentum collected in the trajectory. Note that explicit RMHMC returns a copy of two lists.
    """

    # Cloning params to avoid modification of original variables by in-place operators
    params = params.clone(); momentum = momentum.clone()

    
    # list
    ret_params = []
    ret_momenta = []

    # update 1/2 step momentum (integrator)
    momentum += 0.5 * step_size * params_grad(log_prob_func, params)

    # Leap frog stpes
    for n in range(steps):
        if inv_mass is None:
            params = params + step_size * momentum #/normalizing_const
        else:
            #Assume G is diag here so 1/Mass = G inverse
            if type(inv_mass) is list:
                    i = 0
                    for block in inv_mass:
                        it = block[0].shape[0]
                        params[i:it+i] = params[i:it+i] + step_size * torch.matmul(block,momentum[i:it+i].view(-1,1)).view(-1) #/normalizing_const
                        i += it

            # inv_mass is 2-dimensional matrix (inv_mass(velocity) * momentum(gradient))
            elif len(inv_mass.shape) == 2:
                params = params + step_size * torch.matmul(inv_mass, momentum.view(-1,1)).view(-1) #/normalizing_const

            # inv_mass is vector (inv_mass(velocity) * momentum(gradient))
            else:
                params = params + step_size * inv_mass * momentum #/normalizing_const

        # update gradient
        p_grad = params_grad(log_prob_func, params)

        # update the remaining 1/2 momentum
        momentum += 0.5 * step_size * p_grad

        # Append
        ret_params.append(params.clone())
        ret_momenta.append(momentum.clone())

    # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
    ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
            
    return ret_params, ret_momenta


