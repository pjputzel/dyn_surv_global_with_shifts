import torch
import torch.nn as nn
from loss.DeltaIJBaseLogProbCalculator import DeltaIJBaseLogProbCalculator

class Chen2000LogProbCalculatorDeltaIJ(DeltaIJBaseLogProbCalculator):

    # overwriting to include the clamping
    def compute_shifted_times(self, deltas, batch, eps=1e-20):
        # when t_ij + delta is zero the grad goes to minus infinity
#        deltas.register_hook(clamp_grad_thresh_10) 
#        deltas.register_hook(print_grad)
#        print(deltas)
        shifted_event_times = batch.event_times.unsqueeze(1) + deltas.squeeze(2)

        shifted_cov_times = batch.cov_times + deltas.squeeze(2)
        
        # prevent numerical issues with gradients
        shifted_event_times = shifted_event_times + eps
        shifted_cov_times = shifted_cov_times + eps
#        deltas.register_hook(print_grad)
        #print(deltas, 'deltas')
        #print(torch.max(batch.cov_times, dim=1)[0])
#        print(shifted_cov_times, 'shifted cov time')
        return shifted_event_times, shifted_cov_times

    def compute_logpdf(self, shifted_event_times, global_theta):
        global_theta.register_hook(print_grad)
#        print(global_theta, 'global theta')
        scale = global_theta[0]
        beta = global_theta[1]
        scale.register_hook(clamp_grad)
        beta.register_hook(clamp_grad)
        logpdf = \
            torch.log(beta) + (beta - 1) * torch.log(shifted_event_times)\
            + scale * (1. - torch.exp(shifted_event_times ** beta))\
            + shifted_event_times**beta + torch.log(scale)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        scale = global_theta[0]
        beta = global_theta[1]
        scale.register_hook(clamp_grad)
        beta.register_hook(clamp_grad)
        return scale * (1. - torch.exp(shifted_event_times**beta))
    
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
