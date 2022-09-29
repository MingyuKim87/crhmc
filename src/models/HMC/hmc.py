import os
import sys
# from tqdm import tqmd

# append parent directory
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sampler import *
from .src.gibbs import *
from .src.hamiltonian import *
from .src.leap_frog import *
from .src.acceptance import *

class HMC(Sampler):
    def __init__(self, log_prob_func, param_init, num_samples, leapfrog_steps, step_size, inv_mass, burn=0.0, debug=False):
        self.log_prob_func = log_prob_func
        self.param_init = param_init
        self.num_samples = num_samples
        self.leapfrog_steps = leapfrog_steps
        self.step_size = step_size
        self.inv_mass = inv_mass
        self.burn = burn
        self.debug = debug
        self.store_on_GPU = True if param_init.is_cuda else False
        self.num_rejected = 0
        self.name = "HMC"
        self.n = 0
        
    def append_sample_to_list(self, list, tensor):
        list.append(tensor.clone().detach())
        return list
        
    def draw_samples(self):
        # initialization
        param = self.param_init.clone().requires_grad_()
        device = param.device
        mass = super().mass()
        
        # containers
        ret_params = []

        # Needed for memory moving i.e. move samples to CPU RAM so lookup GPU device
        device = self.param_init.device

        for i in range(self.num_samples):
            # require_grad
            param.requires_grad_()
            
            # sampling
            (param, is_acceptance) = self.sample(
                param,
                self.inv_mass,
                mass
            )
            
            # To DEBUG
                # num_rejected
            self.num_rejected = self.num_rejected + 1 if not is_acceptance is True \
                else self.num_rejected

            # print : 
            if self.debug == 1:
                print('Step: {}, Negative_log_likelihood : {:.2f}'.format(self.n, self.log_prob_func(param).item()))

            # burn
            if self.n >= self.burn:
                ret_params = self.append_sample_to_list(ret_params, param)
                
            # increasing index
            self.n += 1
            
        # type casting to torch.tensor
        ret_params = torch.stack(ret_params, dim=0)

        return ret_params
    
    def sample(self, params, inv_mass, mass):
        """ This is the main sampling function of hamiltorch. Most samplers are built on top of this class. This function receives a function handle log_prob_func,
        which the sampler will use to evaluate the log probability of each sample. A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being
        sampled.

        Parameters
        ----------
        log_prob_func : function
            A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
        params : torch.tensor
            Initialisation of the parameters. This is a vector corresponding to the starting point of the sampler: shape: (D,), where D is the number of parameters of the model.
        leapfrog_steps : int
            The number of steps to take per trajector (often referred to as L).
        step_size : float
            Size of each step to take when doing the numerical integration.
        inv_mass : torch.tensor or list
            The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
            to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
            integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    
        Returns
        -------
        param_samples : list of torch.tensor(s)
            A list of parameter samples. The full trajectory will be returned such that selecting the proposed params requires indexing [1::L] to remove params_innit and select
            the end of the trajectories.
        step_size : float, optional
            Only returned when debug = 2 and using NUTS. This is the final adapted step size.
        acc_rate : float, optional
            Only returned when debug = 2 and not using NUTS. This is the acceptance rate.

        """
        # Needed for memory moving i.e. move samples to CPU RAM so lookup GPU device
        device = params.device

        # momentum
        momentum = gibbs(params, mass=mass)
        
        # hamiltonian
        ham = hamiltonian(params, momentum, self.log_prob_func, inv_mass=inv_mass)
                    
        # Exception treatement
        if has_nan_or_inf(ham):
            raise NotImplementedError

        # HMC
        leapfrog_params, leapfrog_momenta = hmc_leapfrog(params, momentum, self.log_prob_func, steps=self.leapfrog_steps, \
            step_size=self.step_size, inv_mass=inv_mass, store_on_GPU = self.store_on_GPU, debug=self.debug)
                            
        # Exception treatement
        if not len(leapfrog_params) == self.leapfrog_steps:
            raise NotImplementedError
                    
        # candidate
        candidate_params = leapfrog_params[-1].detach().to(device).requires_grad_()
        candidate_momentum = leapfrog_momenta[-1].detach().to(device)
                    
        # new hamiltonian
        new_ham = hamiltonian(candidate_params, candidate_momentum, self.log_prob_func, inv_mass=inv_mass)

        # new_ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric)
        rho = min(0., acceptance(ham, new_ham))
                
        # Initialization
        is_acceptance = None

        # Print            
        if self.debug == 1:
            print('Step: {}, Current Hamiltoninian: {}, Proposed Hamiltoninian: {}'.format(self.n, ham, new_ham))
            
        # Acceptance
        if rho >= torch.log(torch.rand(1)):
            is_acceptance = True 
            print("Accept")
        else:
            is_acceptance = False
            print("Reject")
                    
        return (candidate_params.detach(), is_acceptance) if is_acceptance \
            else (params.detach(), is_acceptance)
            
            
            
            
            
            
            