import os

try:
    import urllib
    from urllib import urlretrieve
except Exception:
    import urllib.request as urllib

import numpy as np
import torch

# mnist
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# iris
from sklearn.datasets import load_iris

def gm_class_data(add_outliers=True, add_class=False):
    """
        Two guassian mixture samples
    """
    
    x0 = np.random.normal(size=50).reshape(-1, 2) - 1
    x1 = np.random.normal(size=50).reshape(-1, 2) + 1.

    x_train, y_train =  np.concatenate([x0, x1]), np.reshape(np.concatenate([np.zeros(25), np.ones(25)]), (-1, 1)).astype(np.int)

    if add_outliers:
        x_1 = np.random.normal(size=10).reshape(-1, 2) + np.array([5., 10.])
        x_train, y_train = np.concatenate([x0, x1, x_1]), np.reshape(np.concatenate([np.zeros(25), np.ones(30)]), (-1, 1)).astype(np.int)
    
    if add_class:
        x2 = np.random.normal(size=50).reshape(-1, 2) + 3.
        x_train, y_train = np.concatenate([x0, x1, x2]), np.reshape(np.concatenate([np.zeros(25), np.ones(25), 2 + np.zeros(25)]), (-1, 1)).astype(np.int)
    
    x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    x_test = np.array([x1_test, x2_test]).reshape(2, -1).T
    
    # containers
    result = {'x_train' : torch.FloatTensor(x_train),\
        'y_train' : torch.FloatTensor(y_train),
        'x_val' : torch.FloatTensor(x_test), \
        'y_val' : None}
    
    return result

def synthetic_sine_data(num_training, num_testing):
    result = {}
    
    result['x_val'] = torch.linspace(-5, 5, num_testing).view(-1,1)
    result['y_val'] = torch.sin(result['x_val']).view(-1,1)

    result['x_train'] = torch.linspace(-3.14, 3.14, num_training).view(-1,1)
    result['y_train'] = torch.sin(result['x_train']).view(-1,1) + torch.randn_like(result['x_train'])*0.1

    return result

def mnist_data(num_training, num_testing, is_all_data=False):
    result = {}
    
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    x_train = mnist_trainset.train_data[:num_training].float()/255. if not is_all_data \
        else mnist_trainset.train_data.float()/255.
    x_val = mnist_trainset.train_data[num_training:num_training+num_testing].float()/255. if not is_all_data \
        else mnist_testset.test_data.float()/255.
    
    result['x_train'] = x_train[:,None] # channel dim
    result['y_train'] = mnist_trainset.train_labels[:num_training].reshape((-1,1)).float() if not is_all_data \
        else mnist_trainset.train_labels.float()/255.
    
    result['x_val'] = x_val[:,None] # channel dim
    result['y_val'] = mnist_testset.test_labels[num_training:num_training+num_testing].reshape((-1,1)).float() if not is_all_data \
        else mnist_testset.test_labels.float()/255.

    return result


def iris_data(num_training, num_testing):
    result = {}
    
    data = load_iris()

    x_ = data['data']
    y_ = data['target']
    
    a = np.arange(x_.shape[0])
    train_index = np.random.choice(a, size = num_training, replace = False)
    val_index = np.delete(a, train_index, axis=0)
    x_train = x_[train_index]
    y_train = y_[train_index]
    x_val = x_[val_index][:]
    y_val = y_[val_index][:]
    x_m = x_train.mean(0)
    x_s = x_train.std(0)
    x_train = (x_train-x_m)/ x_s
    x_val = (x_val-x_m)/ x_s
    D_in = x_train.shape[1]

    result['x_train'] = torch.FloatTensor(x_train)
    result['y_train'] = torch.FloatTensor(y_train)
    result['x_val'] = torch.FloatTensor(x_val)
    result['y_val'] = torch.FloatTensor(y_val)

    return result

def agw_data(num_training, num_testing, get_feats=False, is_all_data=True):
    def features(x):
        """
            x -> (x/2, x**2/4)
            
            Input :
                x : [num_samples]
            Output : 
                features : [num_samples, 2]
        """
        return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0)** 2 ])
    
    
    root = "./data/"
    base_dir = os.path.join(root, "agw_data")

    # download file
    if not os.path.exists(base_dir):
        # mkdir
        os.makedirs(base_dir, exist_ok=True)
        urllib.urlretrieve('https://raw.githubusercontent.com/wjmaddox/drbayes/master/experiments/synthetic_regression/ckpts/data.npy',
                           filename=os.path.join(base_dir,'data.npy'))

    # load data
    data = np.load(os.path.join(base_dir, "data.npy"))

    # split data
    x, y = data[:, 0], data[:, 1]
    y = y[:, None] # expand_dims

    # features 
    f = features(x)

    # normalization
    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)
    f_means, f_stds = f.mean(axis=0), f.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)
    F = ((f - f_means) / f_stds).astype(np.float32)

    # result 
    if get_feats:
        x_train = F[:num_training] if not is_all_data else F
        x_val = F[num_training:] if not is_all_data else None
    else:
        x_train = X[:num_training, None] if not is_all_data else X[:, None]
        x_val = X[num_training:, None] if not is_all_data else np.reshape(np.linspace(-2, 2, 500), (-1, 1))

    y_train = Y[:num_training] if not is_all_data else Y
    y_val = Y[num_training:] if not is_all_data else None

    result = {"x_train" : torch.FloatTensor(x_train),
                "y_train" : torch.FloatTensor(y_train),
                "x_val" : torch.FloatTensor(x_val),
                "y_val" : torch.FloatTensor(y_val)if not y_val == None else torch.empty((1,))}

    return result

if __name__ == "__main__":    
    result = agw_data(100, 50, get_feats=False, is_all_data=True)

    X_test = torch.linspace(-2, 2, 500).view(-1, 1)

    plt.figure(figsize=(12, 5))
    plt.scatter(result['x_train'], result['y_train'])
    plt.grid()
    plt.savefig("./test.png")

    


