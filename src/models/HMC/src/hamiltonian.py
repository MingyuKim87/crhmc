import numpy as np
import torch
import torch.nn as nn
from enum import Enum

from numpy import pi
from .utils import *
from .gibbs import *

def hamiltonian(params, momentum, log_prob_func, inv_mass):
    """Computes the Hamiltonian as a function of the parameters and the momentum.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    jitter : float
        Jitter is often added to the diagonal to the metric tensor to ensure it can be inverted.
        `jitter` is a float corresponding to scale of random draws from a uniform distribution.
    normalizing_const : float
        This constant is currently set to 1.0 and might be removed in future versions as it plays no immediate role.
    softabs_const : float
        Controls the "filtering" strength of the negative eigenvalues. Large values -> absolute value. See Betancourt 2013.
    explicit_binding_const : float
        Only relevant to Explicit RMHMC. Corresponds to the binding term in Cobb et al. 2019.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    ham_func : type
        Only related to semi-separable HMC. This part of hamiltorch has not been fully integrated yet.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.
    integrator : Integrator
        Sets the type of integrator to be used for the leapfrog: Choice {Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING,
        Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
    metric : Metric
        Determines the metric to be used for RMHMC. E.g. default is the Hessian hamiltorch.Metric.HESSIAN.

    Returns
    -------
    torch.tensor
        Returns the value of the Hamiltonian: shape (1,).

    """
    # eval logp
    log_prob = log_prob_func(params)

    # Exceptional Treatement
    if has_nan_or_inf(log_prob):
        print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
                
    # Compute potential energy
    potential = -log_prob 
        
    # For kinetic energy
    if inv_mass is None:
        kinetic = 0.5 * torch.dot(momentum, momentum) #/normalizing_const
        
    # If Inv_mass exists,
    else:
        # When Inv_mass is multiple, 
        if type(inv_mass) is list:
            i = 0
            kinetic = 0
            for block in inv_mass:
                it = block[0].shape[0]
                kinetic = kinetic +  0.5 * torch.matmul(momentum[i:it+i].view(1,-1),torch.matmul(block,momentum[i:it+i].view(-1,1))).view(-1)#/normalizing_const
                i += it
            
        # When Inv_mass is 2-dimensional, 
        elif len(inv_mass.shape) == 2:
            kinetic = 0.5 * torch.matmul(momentum.view(1,-1),torch.matmul(inv_mass,momentum.view(-1,1))).view(-1)#/normalizing_const
                
        # When Inv_mass is a vector, 
        else:
            kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)#/normalizing_const
                
    # Result
    hamiltonian = potential + kinetic
        
    return hamiltonian