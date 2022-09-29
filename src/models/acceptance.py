import numpy as np
import torch
import torch.nn as nn
from enum import Enum

from numpy import pi

from .utils import *
from .properties import *
# cholesky inverse
from .matrix_calculation import *
# fisher information matrix
from .fisher import *

def acceptance(h_old, h_new):
    """Returns the log acceptance ratio for the Metroplis-Hastings step.

    Parameters
    ----------
    h_old : torch.tensor
        Previous value of Hamiltonian (1,).
    h_new : type
        New value of Hamiltonian (1,).

    Returns
    -------
    float
        Log acceptance ratio.

    """
    result = float(-h_new.item() + h_old.item())
    return result

# Adaptation p.15 No-U-Turn samplers Algo 5
def adaptation(rho, t, step_size_init, H_t, eps_bar, desired_accept_rate=0.8):
    """No-U-Turn sampler adaptation of the step size. This follows Algo 5, p. 15 from Hoffman and Gelman 2011.

    Parameters
    ----------
    rho : float
        rho is current acceptance ratio.
    t : int
        Iteration.
    step_size_init : float
        Initial step size.
    H_t : float
        Current rolling H_t.
    eps_bar : type
        Current rolling step size update.
    desired_accept_rate : float
        The step size is adapted with the objective of a desired acceptance rate.

    Returns
    -------
    step_size : float
        Current step size to be used.
    eps_bar : float
        Current rolling step size update. Also at last iteration this is the final adapted step size.
    H_t : float
        Current rolling H_t to be passed at next iteration.

    """
    # rho is current acceptance ratio
    # t is current iteration
    t = t + 1
    if util.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0 # Acceptance rate is zero if nan.
    else:
        alpha = min(1.,float(torch.exp(torch.FloatTensor([rho]))))
    mu = float(torch.log(10*torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1-(1/(t+t0)))*H_t + (1/(t+t0))*(desired_accept_rate - alpha)
    x_new = mu - (t**0.5)/gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t**-kappa * x_new +  (1 - t**-kappa) * torch.log(torch.FloatTensor([eps_bar]))
    eps_bar = float(torch.exp(x_new_bar))

    return step_size, eps_bar, H_t