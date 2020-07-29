import torch
import torch.nn as nn

def print_num_nans(x):
    print(torch.sum(torch.isnan(x) | ~torch.isfinite(x)))

class ThetaIJBaseLogProbCalculator(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    def forward(self, thetas, batch, global_theta):
        cov_times = batch.cov_times
        shifted_event_times = batch.event_times.unsqueeze(1) - cov_times
        logpdf = self.compute_logpdf(shifted_event_times, thetas)
        logsurv = self.compute_logsurv(shifted_event_times, thetas)

        logprob = torch.where(
            batch.censoring_indicators.unsqueeze(1).bool(),
            logsurv, logpdf
        )
        padding_indicators = \
            (batch.cov_times == 0) &\
            torch.cat([torch.zeros(batch.cov_times.shape[0], 1), torch.ones(batch.cov_times.shape[0], batch.cov_times.shape[1] -1 )], dim=1).bool()

        logprob = torch.where(\
            padding_indicators,
            torch.zeros(logprob.shape), logprob
        )

        ret = torch.mean(logprob)
        return ret
    


    def compute_logpdf(self, shifted_event_times, global_theta):
        raise NotImplementedError('Compute logpdf must be defined in subclasses of DeltaIJLogProbCalculator') 

    def compute_logsurv(self, shifted_event_times, global_theta):
        raise NotImplementedError('Compute logsurv must be defined in subclasses of DeltaIJLogProbCalculator') 
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        raise NotImplementedError('Compute lognormalization must be defined in subclasses of DeltaIJLogProbCalculator') 
        
################################################ For cases in [s, s + \delta s] with fixed 
################################################ integration boundaries        
    def compute_cond_prob_over_window(
        self, thetas, batch, global_theta,
        start_time, time_delta
    ):
        # global theta is only in this method sig for
        # convenience, ie for avoiding additional switch statements
        global_theta = None

        max_times_less_than_start, thetas_at_most_recent_time = \
            self.find_most_recent_times_and_thetas(thetas, batch, start_time)

        shifted_end_of_window = \
            start_time + time_delta

        shifted_start_of_window = \
            start_time

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, thetas_at_most_recent_time)
        )
        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, thetas_at_most_recent_time)
        )
        normalization = torch.exp(
            self.compute_logsurv(
                max_times_less_than_start,
                thetas_at_most_recent_time
            )
        )        

        ret = 1./normalization * (start_surv - end_surv)

        return ret

    def find_most_recent_times_and_thetas(self, thetas, batch, start_time):
        max_times_less_than_start, idxs_max_times_less_than_start = \
            batch.get_most_recent_idxs_before_start(start_time)
 
        idxs_thetas = \
            [
                torch.arange(0, idxs_max_times_less_than_start.shape[0]),
                idxs_max_times_less_than_start
            ]
        thetas_at_most_recent_time = thetas[idxs_thetas].squeeze(-1)
        return max_times_less_than_start, thetas_at_most_recent_time

    def compute_most_recent_CDF(
        self, thetas, batch, global_theta,
        start_time, time_delta
    ):

        # for the 'dynamic deephit version' of the c-index with integrating from
        # 0 to S + \Delta S
        max_times_less_than_start, thetas_at_most_recent_time = \
            self.find_most_recent_times_and_thetas(thetas, batch, start_time)


        shifted_end_of_window = \
            start_time + time_delta

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, thetas_at_most_recent_time)
        )

        normalization = torch.exp(
            self.compute_logsurv(
                max_times_less_than_start, 
                thetas_at_most_recent_time
            )
        )        

        ret = 1. - end_surv/normalization 
        return ret

def print_grad(grad):
    print(grad) 
