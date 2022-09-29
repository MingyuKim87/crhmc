import torch

def funnel_ll(w):
    v_dist = torch.distributions.Normal(0,3)
    ll = v_dist.log_prob(w[0])
    x_dist = torch.distributions.Normal(0,torch.exp(-w[0])**0.5)
    ll += x_dist.log_prob(w[1:]).sum()
    return ll


class funnel_loss(object):
    def __init__(self, dimension=10, device='cpu'):
        self.device = device
        self.dimension = dimension

    def log_prob(self, w):
        result = funnel_ll(w)
        return result

    def get_param_init(self):
        D = self.dimension + 1 
        params_init = torch.ones(D)
        params_init[0] = 0.
        return params_init


        
