
import torch
import torch.nn as nn

class RayleighLogProbCalculatorConstantDelta(nn.Module):
    
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
        #deltas.register_hook(print_grad)
        shifted_event_times = batch.event_times + deltas.squeeze(1)

        ### Only need a shift for cov_times if the time isn't zero
        # so for baseline version no shift needed
        shifted_cov_times = batch.cov_times[:, 0]
        #shifted_cov_times = torch.max(batch.cov_times, dim=1)[0] + deltas.squeeze(1)

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
#        pad = torch.zeros(batch.cov_times.shape)
#        cov_times_less_than_start = torch.where(   
#            batch.cov_times < start_time,
#            batch.cov_times, pad
#        )
#        max_times_less_than_start = torch.max(cov_times_less_than_start, dim=1)[0]
        # this is for the baseline t=0 cov only version where t_{ij*} is just 0 for all, 
        # non-baseline (ie the averaging version)  version for constant
        # deltas would use the last event time per person instead


        shifted_max_times_less_than_start = deltas.squeeze(1)
        # again t_{ij*} for model with only covariates at time t=0 is just zeros
        shifted_end_of_window = time_delta + deltas.squeeze(1)

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_max_times_less_than_start, global_theta)
        )


        return 1. - end_surv/start_surv
        
        


def print_grad(grad):
    print(grad) 
