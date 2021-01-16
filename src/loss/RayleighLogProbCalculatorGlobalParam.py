import torch
import torch.nn as nn
from loss.GlobalParamBaseLogProbCalculator import GlobalParamBaseLogProbCalculator
from loss.RayleighLogProbCalculatorDeltaIJ import RayleighLogProbCalculatorDeltaIJ

# This is just a dummy wrapper class for the global param only training.

class RayleighLogProbCalculatorGlobalParam(GlobalParamBaseLogProbCalculator):
    
    def __init__(self, logprob_params):
        super().__init__(logprob_params)
        self.rayleigh = RayleighLogProbCalculatorDeltaIJ(logprob_params)

    def compute_logpdf(self, times, global_theta):
        return self.rayleigh.compute_logpdf(times, global_theta) 

    def compute_logsurv(self, times, global_theta):
        return self.rayleigh.compute_logsurv(times, global_theta)   
 

def print_grad(grad):
    print(grad) 
