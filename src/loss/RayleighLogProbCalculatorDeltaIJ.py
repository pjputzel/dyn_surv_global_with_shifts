import torch
import torch.nn as nn
from loss.DeltaIJBaseLogProbCalculator import DeltaIJBaseLogProbCalculator

class RayleighLogProbCalculatorDeltaIJ(DeltaIJBaseLogProbCalculator):
    def compute_logpdf(self, shifted_event_times, global_theta):
        scale = global_theta[0]
        logpdf = \
            torch.log(shifted_event_times) - torch.log(scale) - \
            shifted_event_times**(2)/(2 * scale)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        scale = global_theta[0]
        return -((shifted_event_times**2)/ (2 * scale))
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        scale = global_theta[0]
        return -((shifted_cov_times**2)/ (2 * scale))


def print_grad(grad):
    print(grad) 
