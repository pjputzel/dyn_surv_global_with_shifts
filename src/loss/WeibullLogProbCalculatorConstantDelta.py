import torch
import torch.nn as nn

class WeibullLogProbCalculatorConstantDelta(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    def forward(self, deltas, batch, global_theta):
        
        shifted_event_times, shifted_cov_times = self.compute_shifted_times(deltas, batch)
        logpdf = self.compute_logpdf(shifted_event_times, global_theta)
        logsurv = self.compute_logsurv(shifted_event_times, global_theta)
        lognormalization = self.compute_lognormalization(shifted_cov_times, global_theta)

        logprob = torch.where(
            batch.censoring_indicators.bool(),
            logsurv, logpdf
        )
        logprob = logprob - lognormalization
        #print(shifted_cov_times[torch.isnan(lognormalization)])
        #logpdf.register_hook(print_grad)
        #logsurv.register_hook(print_grad)
        #lognormalization.register_hook(print_grad)
        #print(shifted_event_times[torch.isnan(logpdf)])
        #logprob.register_hook(print_grad)
        return torch.mean(logprob)
    
    def compute_shifted_times(self, deltas, batch):
        deltas.register_hook(print_grad)
        shifted_event_times = batch.event_times + deltas.squeeze(1)
        shifted_cov_times = torch.max(batch.cov_times, dim=1)[0] + deltas.squeeze(1)
        #deltas.register_hook(print_grad)
        print(deltas, 'deltas')
        #print(torch.max(batch.cov_times, dim=1)[0])
        print(shifted_cov_times, 'shifted cov time')
        return shifted_event_times, shifted_cov_times


    def compute_logpdf(self, shifted_event_times, global_theta):
        global_theta.register_hook(print_grad)
        print(global_theta, 'global theta')
        scale = global_theta[0]
        shape = global_theta[1]
        logpdf = \
            torch.log(shape) - torch.log(scale) + \
            (shape - 1) * (torch.log(shifted_event_times) - torch.log(scale)) - \
            (shifted_event_times/scale)**(shape)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        scale = global_theta[0]
        shape = global_theta[1]
        return -(shifted_event_times/scale)**shape
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        scale = global_theta[0]
        shape = global_theta[1]
        logsurv = -(shifted_cov_times/scale)**(shape)
        return logsurv

def print_grad(grad):
    print(grad) 
