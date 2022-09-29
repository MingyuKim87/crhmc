import torch

def gmm(omega):
        mean = torch.tensor([0.,0.,0.])
        stddev = torch.tensor([.5,1.,2.])
        return torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega).sum()

class gmm_loss(object):
    def __init__(self, mean=torch.tensor([0.,0.,0.]),\
        stddev=torch.tensor([.5, 1., 2.]), device='cpu'):
        self.device = device
        self.mean = mean
        self.stddev = stddev
        self.dimension = self.mean.shape[-1]

    def log_prob(self, w):
        # dimension check
        assert w.shape[-1] == self.mean.shape[-1], "Mismatch dimensionality"
        
        result = gmm(w)
        return result

    def get_param_init(self):
        return torch.zeros(self.dimension)