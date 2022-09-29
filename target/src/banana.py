import numpy as np
import torch

class banana_loss(object):
    def __init__(self, dimension=2, bananicity=3.0, V=0.2, device='cpu'):
        self.device = device
        self.dimension = dimension
        self.bananacity = bananicity
        self.V = V

    def log_prob(self, w):
        transformed_w = torch.zeros_like(w)

        transformed_w[0] = w[0] + self.bananacity * (w[1] ** 2 - self.V)
        # transformed_w[1] = w[1] / np.sqrt(self.V)
        transformed_w[1] = w[1] 
        phi = torch.distributions.Normal(torch.zeros_like(w), torch.ones_like(w))

        result = phi.log_prob(transformed_w).sum()
        return result

    def get_param_init(self):
        D = self.dimension
        
        # assignment (Center)
        # x1 = 0.
        # x2 = self.bananacity * (x1 ** 2 - self.V)
        # params_init = torch.FloatTensor(np.array([x1, x2]))

        # assignemtn
        params_init = torch.rand(self.dimension)

        return params_init


class banana_loss_v2(object):
    def __init__(self, dimension=2, bananicity=3.0, V=0.2, device='cpu'):
        self.device = device
        self.dimension = dimension
        self.bananacity = bananicity
        self.V = V

    def log_prob(self, w):
        
         """Returns U(q), the negative log-posterior formed from the banana distribution"""  
        
         datashift = w[0] + w[1]**2 
         logbanana = 0.5*torch.sum(datashift**2)*4 + 0.5*(w[0]**2 + w[1]**2)
         return -1*logbanana.squeeze()

    def get_param_init(self):
        D = self.dimension
        
        # assignment (Center)
        # x1 = 0.
        # x2 = self.bananacity * (x1 ** 2 - self.V)
        # params_init = torch.FloatTensor(np.array([x1, x2]))

        # assignemtn
        params_init = 0.5*torch.ones(self.dimension)
        params_init[1] = -1 * params_init[1] 

        return params_init
