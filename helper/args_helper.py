# Args parser
import os
import math
from datetime import datetime
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='1D regression')

    common = parser.add_argument_group('common')
    common.add_argument('--sampler', default='HMC', type=str)
    common.add_argument('--metric', default='HESSIAN', type=str)
    common.add_argument('--gpu', default='-1', type=str)
    common.add_argument('--result_root_dir_path', default="./results/", \
         type=str, help='directory path for test data')
    
    args = parser.parse_args()

    return args

# For boolean 
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_dir_path_args(args, dataset_name, **kwargs):
    model_name = args.model

    if kwargs.get("save", None) is not None:
        model_root_path = args.model_root_dir_path
        new_model_root_path = os.path.join(model_root_path, model_name, dataset_name)
        args.model_save_root_dir = new_model_root_path

    return args

def get_result_dir_path_args(args, **kwargs):
    '''
        Make a path for model saving and results
            : args.model_dir_path
    '''
    # set dataset
    dataset = kwargs.get("dataset", None)
    target = kwargs.get("target", "anonymous")
    
    
    if kwargs.get("result_save", None) is True:
        # Generate a unique directory with datatypes, description, current_time
        
        # root dir
        model_root = args.result_root_dir_path

        # Current time
        now = datetime.now()
        current_datetime = str(now.year) + str(now.strftime('%m')) + str(now.strftime('%d'))\
            + str(now.strftime('%H')) + str(now.strftime('%M')) + str(now.strftime('%S'))


        # Make a save(model) directory (Optional)
        result_dir_path = os.path.join(model_root, args.sampler, target, current_datetime) if dataset == None \
            else os.path.join(model_root, args.sampler, target, dataset, current_datetime)
        
        os.makedirs(result_dir_path) if not os.path.isdir(result_dir_path) else None

        return result_dir_path
    
def latest_load_model_filepath(args):
    # Initialize
    filepath = None
    
    # make a file path
    temp_path = os.path.join(args.model_load_dir, args.datatypes, args.description)
    item_list = os.listdir(temp_path)
    item_list = sorted(item_list)
    directory_name = item_list[-1]
    
    load_model_path = os.path.join(args.model_load_dir, args.datatypes, args.description,\
         directory_name, "Model_{}.pt".format(args.datatypes))

    # look for the latest directory and model file ~.pt
    if not os.path.isfile(load_model_path):
        temp_path = os.path.join(args.model_load_dir, args.datatypes, args.description)
        item_list = os.listdir(temp_path)
        item_list = sorted(item_list)

        for item in item_list:
            saved_model_dir = os.path.join(temp_path, item)

            if os.path.isdir(saved_model_dir):
                for f in os.listdir(saved_model_dir):
                    if (f.endswith("pt")):
                        filepath = os.path.join(saved_model_dir, f)
                
        if filepath is None:
            raise NotImplementedError
        else:
            return filepath
            
    else: 
        return load_model_path


def remove_temp_files_and_move_directory(model_dir_path, result_path, *args):
    '''
        Remove temp files and move to /resultMLwM
        
            ./save_models/ --> ./resultMLwM
    '''
    
    temp_path = os.path.join(model_dir_path, "temp")
    file_names = os.listdir(temp_path)

    for filename in file_names:
        if "20" in filename:
            filepath = os.path.join(temp_path, filename)
            os.remove(filepath)

    # Current_time
    now = datetime.now()
    current_datetime = str(now.year) + str(now.month) + str(now.day) + str(now.hour)

    # Naming
    last_path = current_datetime
    for arg in args:
        last_path = last_path + "_" + str(arg)

    # Make a target path    
    result_path = os.path.join(result_path, last_path)

    # copy a temp folder to result folder
    shutil.copytree(model_dir_path, result_path)
    
    # remove temp folder
    shutil.rmtree(model_dir_path)

    print("*"*10, "move the result folder", "*"*10)

    return 0