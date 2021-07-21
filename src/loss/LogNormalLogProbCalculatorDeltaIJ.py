import torch
import numpy as np
import torch.nn as nn
from loss.DeltaIJBaseLogProbCalculator import DeltaIJBaseLogProbCalculator

EPS = 1e-9
class LogNormalLogProbCalculatorDeltaIJ(DeltaIJBaseLogProbCalculator):
    


    def compute_logpdf(self, shifted_event_times, global_theta):
        #global_theta.register_hook(print_grad)
        #print(global_theta, 'global theta')
        mean = global_theta[0]
        scale = global_theta[1]
        #shape = global_theta[1]
        logpdf = \
            torch.log(1/(shifted_event_times * scale * (2*np.pi)**(1/2))) - \
            (torch.log(shifted_event_times  + EPS) - mean)**2 / (2 * scale**2)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        mean = global_theta[0]
        scale = global_theta[1]
        cdf = .5 + .5 * torch.erf((torch.log(shifted_event_times + EPS) - mean)/(2**(1/2) * scale))
        return 1. - cdf
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        return self.compute_logsurv(shifted_cov_times, global_theta)


def print_grad(grad):
    print(grad) 
