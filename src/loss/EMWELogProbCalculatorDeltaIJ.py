
import torch
import torch.nn as nn
from loss.DeltaIJBaseLogProbCalculator import DeltaIJBaseLogProbCalculator

class EMWELogProbCalculatorDeltaIJ(DeltaIJBaseLogProbCalculator):

    # overwriting to include the clamping
    def compute_shifted_times(self, deltas, batch):
        # when t_ij + delta is zero the grad goes to minus infinity
        deltas.register_hook(clamp_grad_tiny_thresh)
#        deltas.register_hook(print_grad)
        shifted_event_times = batch.event_times.unsqueeze(1) + deltas.squeeze(2)

        shifted_cov_times = batch.cov_times + deltas.squeeze(2)

        #deltas.register_hook(print_grad)
        #print(deltas, 'deltas')
        #print(torch.max(batch.cov_times, dim=1)[0])
        #print(shifted_cov_times, 'shifted cov time')
        return shifted_event_times, shifted_cov_times

    def compute_logpdf(self, shifted_event_times, global_theta, upper_clamp=1e8):
        global_theta.register_hook(clamp_grad_tiny_thresh)
        alpha = global_theta[0]
        beta = global_theta[1]
        gamma = global_theta[2]
        lambd = global_theta[3]

        ugly_exponential = torch.exp((shifted_event_times/alpha)**beta)
        logpdf = torch.log(lambd) + torch.log(beta) + torch.log(gamma)\
                 + (beta - 1.) * (torch.log(shifted_event_times) - torch.log(alpha))\
                 + (shifted_event_times/alpha)**beta \
                 + lambd * alpha * (1 - ugly_exponential)\
                 + (gamma - 1.) * torch.log(1. - torch.exp(lambd * alpha * (1. - ugly_exponential)))
        logpdf[~torch.isfinite(logpdf)] = float(upper_clamp)
#        global_theta.register_hook(print_grad)
        #print(global_theta, 'global theta')
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta, eps=1e-8):
        global_theta.register_hook(clamp_grad_tiny_thresh)
        alpha = global_theta[0]
        beta = global_theta[1]
        gamma = global_theta[2]
        lambd = global_theta[3]
        
        ugly_exponential = torch.exp((shifted_event_times/alpha)**beta)
        
        surv = 1. - (1. - torch.exp(lambd * alpha * (1. - ugly_exponential)) )**gamma
        surv[surv == 0] = surv[surv == 0] + float(eps)
#        if surv == 0:
#            surv = surv + float(eps)
        return torch.log(surv)
    
    def compute_lognormalization(self, shifted_cov_times, global_theta):
        return self.compute_logsurv(shifted_cov_times, global_theta)
        


def print_grad(grad):
    print(grad)

def clamp_grad(grad, thresh=1.):
    grad[grad > float(thresh)] = thresh
    # for beta which goes to positive infinity
    grad[torch.isnan(grad)] = thresh
    grad[grad < float(-thresh)] = -thresh

def clamp_grad_thresh_10(grad):
    clamp_grad(grad, thresh=10.)

def clamp_grad_tiny_thresh(grad):
    clamp_grad(grad, thresh=.5)
