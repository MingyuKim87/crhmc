from abc import *
import torch

class Sampler(metaclass=ABCMeta):
    def __init__(self, log_prob_func, param_init, num_samples, leapfrog_steps, step_size, inv_mass, burn, debug, store_on_GPU):
        self.log_prob_func = log_prob_func
        self.param_init = param_init
        self.num_samples = num_samples
        self.leapfrog_steps = leapfrog_steps
        self.step_size = step_size
        self.inv_mass = inv_mass
        self.burn = burn
        self.debug = debug
        self.store_on_GPU = store_on_GPU
    
    def __call__(self):
        # Exceptional treatment
        if self.param_init.dim() != 1:
            raise RuntimeError('params_init must be a 1d tensor.')

        if self.burn >= self.num_samples:
            raise RuntimeError('burn must be less than num_samples.')
        
    def draw_samples(self):
        pass
    
    def mass(self):
        # Mass 
        # Invert mass matrix once (As mass is used in Gibbs resampling step)
        mass = None
        if self.inv_mass is not None:
            if type(self.inv_mass) is list:
                mass = []
                for block in self.inv_mass:
                    mass.append(torch.inverse(block))
            #Assum G is diag here so 1/Mass = G inverse
            elif len(self.inv_mass.shape) == 2:
                mass = torch.inverse(self.inv_mass)
            elif len(self.inv_mass.shape) == 1:
                mass = 1/self.inv_mass
                
        return mass
    
    @abstractmethod
    def draw_samples(self):
        pass
    

        