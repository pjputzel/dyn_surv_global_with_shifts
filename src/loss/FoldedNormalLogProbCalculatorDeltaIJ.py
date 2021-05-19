import torch
import math
import torch.nn as nn
from loss.DeltaIJBaseLogProbCalculator import DeltaIJBaseLogProbCalculator

class FoldedNormalLogProbCalculatorDeltaIJ(DeltaIJBaseLogProbCalculator):

    def compute_logpdf(self, shifted_event_times, global_theta):
        mu = global_theta[0]
        sigma = global_theta[1]
        t1 =\
            (
                1/(sigma * (2 * math.pi)**(1./2.)) * \
                torch.exp(-(shifted_event_times - mu)**2/(2 * sigma**2))
            )
        t2 =\
            (
                1/(sigma * (2 * math.pi)**(1./2.)) * \
                torch.exp(-(shifted_event_times + mu)**2/(2 * sigma**2))
            ) 
            
        logpdf = torch.log(
            (
                1/(sigma * (2 * math.pi)**(1./2.)) * \
                torch.exp(-(shifted_event_times - mu)**2/(2 * sigma**2))
            ) +\
            (
                1/(sigma * (2 * math.pi)**(1./2.)) * \
                torch.exp(-(shifted_event_times + mu)**2/(2 * sigma**2))
            ) 
            
        )
#        print(shifted_event_times, mu, sigma)
#        print(t1)
#        print(t2)
#        print(logpdf)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        mu = global_theta[0]
        sigma = global_theta[1]
#        logsurv = -eta * (torch.exp(b * shifted_event_times) - 1)
#        print(logsurv)
        logsurv = 0.5 * (\
            torch.erf((shifted_event_times + mu)/(sigma * 2**(1/2))) +\
            torch.erf((shifted_event_times - mu)/(sigma * 2**(1/2)))
        )
#        print(logsurv)
#        raise RuntimeError('Preston stopped the code')
        return logsurv
    
    def compute_lognormalization(self, shifted_cov_times, global_theta):
        return self.compute_logsurv(shifted_cov_times, global_theta)
        


def print_grad(grad):
    print(grad)

def clamp_grad(grad, thresh=5.):
    grad[grad > float(thresh)] = thresh
    # for beta which goes to positive infinity
    grad[torch.isnan(grad)] = -thresh
    grad[grad < float(-thresh)] = -thresh

def clamp_grad_thresh_10(grad):
    clamp_grad(grad, thresh=10.)
