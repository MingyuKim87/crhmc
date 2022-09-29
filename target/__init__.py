import torch

from .src.funnel import funnel_loss
from .src.gmm import gmm_loss
from .src.bnn import bnn_loss
from .src.banana import banana_loss, banana_loss_v2
from .src.bayes_logit_regress import bayes_logit_loss
from .src.bnn_utils.net import Net, Conv_Net

# dataset
from .src.bnn_utils.dataset import *

def select_dataset(kargs):
    # loss type
    loss_type = kargs.get("loss_type")
    
    # dataset
    dataset = kargs.get("dataset", None)
    num_tr = kargs.get("num_tr", None)
    num_val = kargs.get("num_val", None)

    # Exceptional treatment
    if dataset == None:
        raise NotImplementedError()
    
    if loss_type == "regression" and dataset == "sine":
        # dataset 
        dataset = synthetic_sine_data(num_tr, num_val)

    elif loss_type == "regression" and dataset == "agw":
        # dataset 
        dataset = agw_data(num_tr, num_val, is_all_data=True)

    elif loss_type == "binary_class_linear_output" and dataset == "gm":
        # dataset 
        dataset = gm_class_data()
    
    elif loss_type == "multi_class_linear_output" and dataset == "iris":
        # dataset 
        dataset = iris_data(num_tr, num_val)

    elif loss_type == "multi_class_linear_output" and dataset == "mnist":
        # dataset 
        dataset = mnist_data(num_tr, num_val, is_all_data=True)
    else:
        dataset = None

    return dataset

def select_architecture(layer_sizes, loss_type, kargs):
    if not kargs.get("is_conv_net", False):
        net = Net(layer_sizes=layer_sizes, loss=loss_type)
    else:
        net = Conv_Net(loss=loss_type)

    return net

def get_target_dist(**kargs):
    # get properties of bnn
    layer_sizes = kargs.get("layer_sizes")
    loss_type = kargs.get("loss_type")
    device = kargs.get("device", 'cpu')

    # prior scale
    prior_scale = kargs.get("prior_scale", 1.0)

    # dataset (Initialization)
    dataset = None

    if layer_sizes == None and loss_type == None:
        if kargs.get("target") == "gmm":
            target_dist = gmm_loss()
        elif kargs.get("target") == "funnel":
            target_dist = funnel_loss()
        elif kargs.get("target") == "banana":
            target_dist = banana_loss_v2()
        else:
            raise NotImplementedError()

    elif kargs.get("target") == "bnn":
        # net architecture
        net = select_architecture(layer_sizes=layer_sizes, loss_type=loss_type, kargs=kargs)
        
        # loss function
        target_dist = bnn_loss(net, prior_scale, device)

        # dataset
        dataset = select_dataset(kargs)
    elif kargs.get("target") == "blr":
        # net architecture
        net = select_architecture(layer_sizes=layer_sizes, loss_type=loss_type, kargs=kargs)
        
        # loss function
        target_dist = bayes_logit_loss(net, prior_scale, device)

        # dataset
        dataset = select_dataset(kargs)
    else:
        raise NotImplementedError()

    # set dataset
    if not dataset == None:
        target_dist.set_train_dataset(dataset['x_train'], dataset['y_train'])
        target_dist.set_test_dataset(dataset['x_val'], dataset['y_val'])

        # set prior distribution : set tau_list
        target_dist.set_tau()
    
    return target_dist



