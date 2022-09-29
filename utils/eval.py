import numpy as np
import torch
import pymc3

def log_likelihood(pred_y, y):
    """
        Input : 
            pred_y : [num_samples, num_y]
            y : [num_y]

        Output:
            mse : scalar
    """

    mean_function = np.mean(pred_y, axis=0)
    std_function = np.std(pred_y, axis=0)

    # torch distribution
    normal_dist = torch.distributions.normal.Normal(torch.FloatTensor(mean_function), torch.FloatTensor(std_function))

    # log_prob
    log_prob = normal_dist.log_prob(torch.FloatTensor(y))

    # Average
    result = log_prob.mean().item()

    return result


def mse(pred_y, y):
    """
        Input : 
            pred_y : [num_samples, num_y]
            y : [num_y]

        Output:
            mse : scalar
    """
    mse = np.mean((np.mean(pred_y, axis=0) - y)**2)
    return mse


def minESS(samples, burn = 50):
    # container
    result = []

    # slicing (burn)
    filtered_samples = samples[burn:]

    for d in range(samples.shape[-1]):
        # slicing dimensionality
        sample = filtered_samples[:, d]

        # calculating ess
        temp_ess = pymc3.ess(sample)

        # append
        result.append(temp_ess)

    return min(result)