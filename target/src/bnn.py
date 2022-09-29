from sklearn.datasets import load_iris
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

# from target.src.bnn_utils.utils import make_functional

# bnn utils
from .bnn_utils.dataset import *
from .bnn_utils.utils import *

class bnn_loss(object):
    def __init__(self, net, prior_scale=1.0, device='cpu'):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.device = device
        self.net = net.to(self.device)
        self.tau_list = self.set_tau()
        self.prior_scale = prior_scale
        

    def set_train_dataset(self, x_train, y_train):
        self.x_train = x_train.to(self.device)
        self.y_train = y_train.to(self.device)
        
    def set_test_dataset(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val
        
    def set_networks(self, layer_sizes, loss = "multi_class", bias=True):
        net = Net(layer_sizes, loss = "multi_class", bias=True)
        return net
    
    def set_tau(self, tau_list=None):
        if tau_list == None:
            tau_list_default = []
            tau = 1.
        
            for w in self.net.parameters():
                tau_list_default.append(tau)
            
            tau_list_default = torch.tensor(tau_list_default).to(self.device)
        else:
            tau_list_default = tau_list.to(self.device)
        
        return tau_list_default

    def get_tau(self):
        return self.tau_list

    def flatten(self):
        result = torch.cat([p.flatten() for p in self.net.parameters()])
        return result
    
    def get_param_init(self):
        return self.flatten()
    
    def get_net_params(self):
        params_shape_list = []
        params_flattened_list = []
        
        for weights in self.net.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            
        return params_shape_list, params_flattened_list

    def get_loss(self, output, y_train, tau_out):
        if not (torch.is_tensor(output) and torch.is_tensor(y_train)):
            output = torch.Tensor(output)
            y_train = torch.Tensor(y_train)
        
        # to device
        output = output.to(self.device)
        y_train = y_train.to(self.device)
        
        # model_loss
        if self.net.loss == "binary_class_linear_output":
            loss = nn.BCEWithLogitsLoss(reduction='sum')
            ll = -1*tau_out * (loss(output, y_train))

        elif self.net.loss == "multi_class_linear_output":
            loss = nn.CrossEntropyLoss(reduction='sum')
            ll = -1*tau_out * (loss(output, y_train.long().view(-1)))

        elif self.net.loss == "multi_class_log_softmax_output":
            ll = -1*tau_out * (torch.nn.functional.nll_loss(output, y_train.long().view(-1)))

        elif self.net.loss == "regression":
            ll = -0.5 * tau_out * ((output-y_train)**2).sum(0)

        else:
            raise NotImplementedError()

        return ll

    def unflatten(self, params):
        '''
            Input:
                params : flattened_list (torch.tensor)
            Output:
                unflattend params : unflattened_list (list)
        '''
        
        assert params.dim() == 1, "Expecting a 1D flattened_params"
        
        # index
        i = 0

        # container
        result = []
        
        # unflatten
        for layer in list(self.net.parameters()):
            n_element = layer.nelement()
            param = params[i:i+n_element].view_as(layer)
            result.append(param)
            i += n_element

        return result

    def log_prob(self, tau_out=1., normalizing_const=1., is_predict= False):
        # net shape list
        params_shape_list, params_flattened_list = self.get_net_params()

        # tau_list
        dist_list = []
        for tau in self.tau_list:
            dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))

        def func_log_prob(params):
            # functional models 
            f_net = make_functional(self.net)
            
            # params
            unflattened_params = self.unflatten(params)

            # prior
            prior_likelihood = torch.zeros(1).to(self.device) # initialization
            for layer, prior_dist in zip(unflattened_params, dist_list):
                prior_likelihood += prior_dist.log_prob(layer).sum() # diagonal normal N(0,I) 

            # set device
            x_train = self.x_train.to(self.device)
            y_train = self.y_train.to(self.device)

            # feed-forward
            output = f_net(x_train, params=unflattened_params)

            # model_loss
            ll = self.get_loss(output, y_train, tau_out)

            # memory detach
            if self.device != "cpu":
                del x_train, y_train
                torch.cuda.empty_cache()

            if is_predict:
                return (ll + prior_likelihood/self.prior_scale), output

            else:
                return (ll + prior_likelihood/self.prior_scale)

        return func_log_prob

    def test_log_prob(self, tau_out=1.):
        with torch.no_grad():
            # net shape list
            params_shape_list, params_flattened_list = self.get_net_params()

            # tau_list
            dist_list = []
            for tau in self.tau_list:
                dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))

            def func_log_prob(params):
                # functional models 
                f_net = make_functional(self.net)

                # params
                unflattened_params = self.unflatten(params)

                # prior
                prior_likelihood = torch.zeros(1) # initialization
                for layer, prior_dist in zip(unflattened_params, dist_list):
                    prior_likelihood += prior_dist.log_prob(layer).sum() # diagonal normal N(0,I) 

                # set device
                x_test = self.x_test.to(self.device)
                y_test = self.y_test.to(self.device)

                # feed-forward
                output = f_net(x_test, params=unflattened_params)

                # model_loss
                ll = self.get_loss(output, y_test, tau_out)

                # memory detach
                if self.device != "cpu":
                    del x_device, y_device
                    torch.cuda.empty_cache()

                
                return (ll + prior_likelihood/self.prior_scale), output

    def forward(self, x, sample):
        '''
            Input
                x : feature values
                sample : network parameters (weights)
    
            Output:
                pred_y : predicted values            
        '''
        # to device
        x = x.to(self.device)
        param = sample.to(self.device)

        # unflatten params
        unflattened_params = self.unflatten(param)

        # functional models 
        f_net = make_functional(self.net)

        # feed-forward
        output = f_net(x, params=unflattened_params)

        return output


                

            
            





    
    
    