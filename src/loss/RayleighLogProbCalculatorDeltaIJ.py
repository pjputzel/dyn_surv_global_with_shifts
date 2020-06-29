import torch
import torch.nn as nn

class RayleighLogProbCalculatorDeltaIJ(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    def forward(self, deltas, batch, global_theta):
        
        shifted_event_times, shifted_cov_times = self.compute_shifted_times(deltas, batch)
        logpdf = self.compute_logpdf(shifted_event_times, global_theta)
        logsurv = self.compute_logsurv(shifted_event_times, global_theta)
        lognormalization = self.compute_lognormalization(shifted_cov_times, global_theta)

#        print(logpdf.shape, logsurv.shape, batch.censoring_indicators.shape, lognormalization.shape)

        logprob = torch.where(
            batch.censoring_indicators.unsqueeze(1).bool(),
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
        #deltas.register_hook(print_grad)
        shifted_event_times = batch.event_times.unsqueeze(1) + deltas.squeeze(2)
        

        shifted_cov_times = batch.cov_times + deltas.squeeze(2)

        #deltas.register_hook(print_grad)
        #print(deltas, 'deltas')
        #print(torch.max(batch.cov_times, dim=1)[0])
        #print(shifted_cov_times, 'shifted cov time')
        return shifted_event_times, shifted_cov_times


    def compute_logpdf(self, shifted_event_times, global_theta):
        #global_theta.register_hook(print_grad)
        #print(global_theta, 'global theta')
        scale = global_theta[0]
        #shape = global_theta[1]
        logpdf = \
            torch.log(shifted_event_times) - torch.log(scale) - \
            shifted_event_times**(2)/(2 * scale)
        return logpdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        scale = global_theta[0]
        return -(shifted_event_times/scale)
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        scale = global_theta[0]
        logsurv = -(shifted_cov_times/scale)
        return logsurv

    def compute_cond_prob_over_window(
        self, deltas, batch, global_theta,
        start_time, time_delta
    ):
        # first find j*
        pad = torch.zeros(batch.cov_times.shape)
        cov_times_less_than_start = torch.where(   
            batch.cov_times < start_time,
            batch.cov_times, pad
        )
        max_times_less_than_start = torch.max(cov_times_less_than_start, dim=1)[0] 
        
        
        # TODO: update so this entire function works in general
        shifted_end_of_window = start_time + time_delta + deltas.squeeze(1)
        shifted_start_of_window = start_time + deltas.squeeze(1)

        #print(shifted_start_of_window)

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, global_theta)
        )
        
        cond_surv_j = torch.exp(
            self.compute_logsurv(max_times_less_than_start, global_theta)
        )
        print(start_surv - end_surv)
        ret = (1./cond_surv_j) * (start_surv - end_surv)

        return ret
        
        


def print_grad(grad):
    print(grad) 
