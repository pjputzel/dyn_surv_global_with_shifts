
import torch
import torch.nn as nn
from loss.ThetaIJBaseLogProbCalculator import ThetaIJBaseLogProbCalculator

class RayleighLogProbCalculatorThetaIJ(ThetaIJBaseLogProbCalculator):
    


    def compute_logpdf(self, shifted_event_times, thetas):
#        thetas.register_hook(print_grad)
#        print(thetas, 'thetas')
        scale = thetas[:, 0]
        #shape = global_theta[1]
        logpdf = \
            torch.log(shifted_event_times) - torch.log(scale) - \
            shifted_event_times**(2)/(2 * scale)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, thetas):
        if not len(thetas.shape) == 1:
            scale = thetas[:, 0]
        else:
            scale = thetas
        return -((shifted_event_times**2)/ (2 * scale))
    
    def compute_lognormalization(self, shifted_cov_times, thetas):
        return self.compute_logsurv(shifted_cov_times, thetas)


def print_grad(grad):
    print(grad) 
