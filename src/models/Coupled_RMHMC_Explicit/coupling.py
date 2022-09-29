import math
import random
import numpy as np
import torch
from .utils import *

def maximalcoupling(energy_1, energy_2):
    mu = energy_1 / energy_1.sum()
    nu = energy_2 / energy_2.sum()

    TV_distance = torch.abs(mu - nu)
    TV_distance = TV_distance.max().item()

    omega = 1.0 - TV_distance

    # minimum
    pqmin = torch.minimum(mu, nu)
    Z = pqmin.sum()

    # random sampling
    u = random.random()

    if u < omega:
        prob = pqmin / Z
        
        print(prob)
                
        if not torch.all(prob > 0.0).item():
            raise Exception("softmax prob dose not match sum to 1")
        
        # define distribution
        dist = torch.distributions.categorical.Categorical(pqmin/Z)

        # sampling
        i = dist.sample().long().item()
        j = i

    else:
        # define distribution
        dist_1 = torch.distributions.categorical.Categorical((mu - pqmin)/Z)
        dist_2 = torch.distributions.categorical.Categorical((nu - pqmin)/Z)

        # samplings
        i = dist_1.sample().long().item() 
        j = dist_2.sample().long().item()

    # DEBUG
    print("i : ", i)
    print("j : ", j)
    
    return (i, j)


def totalvariation_distance(mu, nu):
    '''
        For categorial distribution
    '''
    result = torch.abs(mu - nu).max
    return result

if __name__ == "__main__":
    a = torch.tensor([1,1,1])
    b = torch.tensor([2,2,2])
    c = torch.vstack((a,b))
    

    print(torch.minimum(a, b))




