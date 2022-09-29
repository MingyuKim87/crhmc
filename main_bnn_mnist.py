import os
import sys
import time
from datetime import datetime
import yaml

import numpy as np
import matplotlib.pyplot as plt

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# yml
import yaml

# utils
import utils

# model src
from src.models import REGISTRY as sampler_registry

# target
from target import *

# arg_parse
import helper

def get_device(device):
    """
        Input :
            device : str
        Output : 
            device : torch.device
    """
    # type casting
    device_num = float(device)
    
    if device_num >= 0:
        device_num = int(float(device))
        torch.cuda.set_device(device_num)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

def run():
    # Parser
    args = helper.parse_args()

    # device
    device = get_device(args.gpu)

    # config
    config = yaml.load(open("./config/envs_mnist.yml", 'r'), Loader=yaml.SafeLoader)
    
    # get result paths
    result_dir = helper.get_result_dir_path_args(args, dataset=config["BNN_DATASET"], target=config["BASEMODEL"], result_save=True)
    
    # get model properties
    metric_properties = utils.get_metric_properties(args)
    
    # Set seed
    utils.set_random_seed(config['SEED'])
    
    # get target dist and param_init
    target_dist = get_target_dist(layer_sizes=config["BNN_LAYER_SIZES"], 
        loss_type=config["BNN_LOSS_TYPE"],
        dataset=config["BNN_DATASET"],
        num_tr=config["BNN_NUM_TRAIN"],
        num_val=config["BNN_NUM_VAL"],
        target=config["BASEMODEL"],
        is_conv_net=True,
        device = device
    )

    func_target_dist = target_dist.log_prob(tau_out=config["TAU_OUT"])
    params_init = target_dist.get_param_init()

    # process time
    start = time.process_time()
    
    # open stdout
    if not config['DEBUG']:
        sys.stdout = open(os.path.join(result_dir, 'stdout.txt'), 'w')
        
    # Sampler
    HMC = sampler_registry[args.sampler] # generator
    HMC_sampler = HMC(log_prob_func=func_target_dist,
        param_init=params_init,
        num_samples=config['NUM_SAMPLES'],
        leapfrog_steps=config['LEAP_STEP'],
        step_size=config['STEP_SIZE'],
        inv_mass = torch.ones_like(params_init),
        debug=config['DEBUG']
    )
            
    # samples
    sampled_params = HMC_sampler.draw_samples()
    
    # save config 
    json.dump(config, open(os.path.join(result_dir, "config.json"), 'w'))
        
    # save torch file
    torch.save(sampled_params, os.path.join(result_dir, "result_RMHMC.pt"))
    
    # save config
    json.dump(config, open(os.path.join(result_dir, "config.json"), 'w'))
    
    # container
    pred_list, log_prob_list = [], []
    
    # Feed forward
    for param in sampled_params:
        pred_y = target_dist.forward(target_dist.x_val, param)
        log_prob = target_dist.get_loss(pred_y, target_dist.y_val, tau_out=config["TAU_OUT_VAL"])

        pred_list.append(pred_y.detach().cpu().numpy())
        log_prob_list.append(log_prob.detach().cpu().numpy())

    # Print 
    print('\nmin ESS: {:.2f}'.format(utils.minESS(sampled_params.cpu().numpy(), burn=0)))

    # save results
    yaml.dump({"minESS" : "{:.2f}".format(utils.minESS(sampled_params.cpu().numpy(), burn=0))}, 
        open(os.path.join(result_dir, "result.yml"), 'w'))

    # training curves
        # container
    pred_list_tr, log_prob_list_tr = [], []

        # Feed forward (training set)
    for param in sampled_params:
        pred_y_tr = target_dist.forward(target_dist.x_train, param)
        log_prob_tr = target_dist.get_loss(pred_y_tr, target_dist.y_train, tau_out=config["TAU_OUT_VAL"])

        pred_list_tr.append(pred_y_tr.cpu().numpy())
        log_prob_list_tr.append(log_prob_tr.cpu().numpy())

    # plot
    utils.plot_bnn_learning_curves(
        target_dist.get_loss,\
        pred_list_tr, \
        target_dist.y_train.cpu().numpy(), tau_out=config["TAU_OUT_VAL"], \
        file_name=os.path.join(result_dir, "bnn_training_curves_" + datetime.today().strftime("%Y%m%d%H%M")) + ".png")
    
    # close stdout
    if not config['DEBUG']:
        sys.stdout.close()
    
if __name__ == "__main__":    
    # run
    run()

    # delete all auxiliary files
    os.system("find . | grep -E \"(__pycache__|\.pyc|\.pyo$)\" | xargs rm -rf")

    
    