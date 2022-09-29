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
    
    # get model properties
    metric_properties = utils.get_metric_properties(args)
    
    # config
    config = yaml.load(open("./config/envs_banana.yml", 'r'), Loader=yaml.SafeLoader)

    # get result paths
    result_dir = helper.get_result_dir_path_args(args, target=config["BASEMODEL"], result_save=True)
    
    # Set seed
    utils.set_random_seed(config['SEED'])
    
    # get target dist and param_init
    target_dist = get_target_dist(target=config["BASEMODEL"])
    func_target_dist = target_dist.log_prob
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
        inv_mass = None,
        debug=config['DEBUG'],
    )
            
    # samples
    sampled_params = HMC_sampler.draw_samples()
    
    # save config 
    json.dump(config, open(os.path.join(result_dir, "config.json"), 'w'))
        
    # save torch file
    torch.save(sampled_params, os.path.join(result_dir, "result_RMHMC.pt"))
    
    # save config5
    json.dump(config, open(os.path.join(result_dir, "config.json"), 'w'))
    
    # plot
    utils.plot_banana_dist(func_target_dist, sampled_params, os.path.join(result_dir, "banana_result_coords" + datetime.today().strftime("%Y%m%d%H%M")) + ".png")
    
    # close stdout
    if not config['DEBUG']:
        sys.stdout.close()


    
if __name__ == "__main__":    
    run()

    # delete all auxiliary files
    os.system("find . | grep -E \"(__pycache__|\.pyc|\.pyo$)\" | xargs rm -rf")
    
    