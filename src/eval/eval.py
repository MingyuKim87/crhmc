import torch

class evalation(object):
    def __init__(self):
        pass
    
    def gmm_eval(self, solution_mean, solution_stddev, sample_results):
        distributions = []
        solution_distribution = torch.distributions.MultivariateNormal(solution_mean, solution_stddev.diag()**2)
        result = dict()
        
        for key, item in sample_results.items():
            distribution = torch.distributions.MultivariateNormal(item.mean(dim=0), torch.diag(torch.FloatTensor(item.var(dim=0))))
            result[key] = torch.distributions.kl.kl_divergence(solution_distribution, distribution)
            
        print(result)
        return result
            
        