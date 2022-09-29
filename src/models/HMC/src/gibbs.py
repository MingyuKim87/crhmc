import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from numpy import pi


def gibbs(params, mass=None):
    """Performs the momentum resampling component of HMC.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    jitter : float
        Jitter is often added to the diagonal to the metric tensor to ensure it can be inverted. `jitter` is a float corresponding to scale of random draws from a uniform distribution.
    normalizing_const : float
        This constant is currently set to 1.0 and might be removed in future versions as it plays no immediate role.
    softabs_const : float
        Controls the "filtering" strength of the negative eigenvalues. Large values -> absolute value. See Betancourt 2013.
    mass : torch.tensor or list
        The mass matrix is related to the inverse covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of  shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the mass matrix as a list of blocks. Hope to make that more efficient.
    metric : Metric
        Determines the metric to be used for RMHMC. E.g. default is the Hessian hamiltorch.Metric.HESSIAN.

    Returns
    -------
    torch.tensor
        Returns the resampled momentum vector of shape (D,).

    """    
    if mass is None:
        # If mass is None, dist ~ N(0, I)
        dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    
    else:
        if type(mass) is list:
            # block wise mass list of blocks
            samples = torch.zeros_like(params)
            i = 0
            for block in mass:
                it = block[0].shape[0]
                dist = torch.distributions.MultivariateNormal(torch.zeros_like(block[0]), block)
                samples[i:it+i] = dist.sample()
                i += it
            return samples
        elif len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), mass)
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass ** 0.5) # Normal expects standard deviation so need sqrt
    return dist.sample()

