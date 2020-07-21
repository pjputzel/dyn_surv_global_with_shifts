
import torch
import torch.nn as nn

class WeibullLogProbCalculatorDeltaIJ(nn.Module):
    
    def compute_shifted_times(self, deltas, batch):
        #deltas.register_hook(print_grad)
        deltas.register_hook(clamp_grad)
        shifted_event_times = batch.event_times.unsqueeze(1) + deltas.squeeze(2)
        

        shifted_cov_times = batch.cov_times + deltas.squeeze(2)

        #deltas.register_hook(print_grad)
        #print(deltas, 'deltas')
        #print(torch.max(batch.cov_times, dim=1)[0])
        #print(shifted_cov_times, 'shifted cov time')
        return shifted_event_times, shifted_cov_times

    def compute_logpdf(self, shifted_event_times, global_theta):
#        global_theta.register_hook(print_grad)
#        print(global_theta, 'global theta')
        global_theta.register_hook(clamp_grad)
        scale = global_theta[0]
        shape = global_theta[1]
        logpdf = \
            torch.log(shape) - torch.log(scale) + \
            (shape - 1) * (torch.log(shifted_event_times) - torch.log(scale)) - \
            (shifted_event_times/scale)**(shape)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        global_theta.register_hook(clamp_grad)
        scale = global_theta[0]
        shape = global_theta[1]
        return -(shifted_event_times/scale)**shape
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        scale = global_theta[0]
        shape = global_theta[1]
        logsurv = -(shifted_cov_times/scale)**(shape)
        return logsurv


def print_grad(grad):
    print(grad, torch.sum(torch.isnan(grad)))

def clamp_grad(grad, thresh=5.):
    grad[grad > float(thresh)] = thresh
    # for beta which goes to positive infinity
    grad[torch.isnan(grad)] = thresh
    grad[grad < float(-thresh)] = -thresh
