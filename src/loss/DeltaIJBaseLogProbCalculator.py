import torch
import torch.nn as nn

def print_num_nans(x):
    print(torch.sum(torch.isnan(x) | ~torch.isfinite(x)))

class DeltaIJBaseLogProbCalculator(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    def forward(self, deltas, batch, global_theta):
        
        shifted_event_times, shifted_cov_times = self.compute_shifted_times(deltas, batch)
        logpdf = self.compute_logpdf(shifted_event_times, global_theta)
        logsurv = self.compute_logsurv(shifted_event_times, global_theta)
        lognormalization = self.compute_lognormalization(shifted_cov_times, global_theta)
#        print_num_nans(logpdf)
#        print_num_nans(logsurv)
#        print_num_nans(lognormalization)
#        print(logpdf.shape, logsurv.shape, batch.censoring_indicators.shape, lognormalization.shape)

        logprob = torch.where(
            batch.censoring_indicators.unsqueeze(1).bool(),
            logsurv, logpdf
        )
#        print('nans before and after subtraction:')
#        print_num_nans(logprob)
        logprob = logprob - lognormalization
#        print_num_nans(logprob)
        # zero out the padded values
        #print(batch.cov_times[:, 200])
        padding_indicators = \
            (batch.cov_times == 0) &\
            torch.cat([torch.zeros(batch.cov_times.shape[0], 1), torch.ones(batch.cov_times.shape[0], batch.cov_times.shape[1] -1 )], dim=1).bool()
        logprob = torch.where(\
            padding_indicators,
            torch.zeros(logprob.shape), logprob
        )
#        print(deltas[torch.isnan(logprob)])
#        print(lognormalization[torch.isnan(logprob)])
#        print(logsurv[torch.isnan(logprob)])
#        print(logpdf[torch.isnan(logprob)])
#        print(torch.sum(torch.isnan(logprob)))
        #print(logprob[:, 200])
        #print(shifted_cov_times[torch.isnan(lognormalization)])
        #logpdf.register_hook(print_grad)
        #logsurv.register_hook(print_grad)
        #lognormalization.register_hook(print_grad)
        #print(shifted_event_times[torch.isnan(logpdf)])
        #logprob.register_hook(print_grad)
        ret = torch.mean(logprob)
        #print(ret.shape)
        return ret
    
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
        raise NotImplementedError('Compute logpdf must be defined in subclasses of DeltaIJLogProbCalculator') 

    def compute_logsurv(self, shifted_event_times, global_theta):
        raise NotImplementedError('Compute logsurv must be defined in subclasses of DeltaIJLogProbCalculator') 
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        raise NotImplementedError('Compute lognormalization must be defined in subclasses of DeltaIJLogProbCalculator') 
########################################### For cases in [t_{ij*}, t_{ij*} + \delta p] version with
########################################### variable integration bounds
#    def compute_cond_prob_over_window(
#        self, deltas, batch, global_theta,
#        start_time, time_delta
#    ):
#        # first find j*
#        pad = torch.zeros(batch.cov_times.shape)
#        cov_times_less_than_start = torch.where(   
#            batch.cov_times < start_time,
#            batch.cov_times, pad
#        )
#        max_times_less_than_start, idxs_max_times_less_than_start = \
#            torch.max(cov_times_less_than_start, dim=1)      
# 
##        print(deltas.shape, deltas.squeeze(-1)[:, idxs_max_times_less_than_start].shape, max_times_less_than_start.shape)
#        idxs_deltas = \
#            [
#                torch.arange(0, idxs_max_times_less_than_start.shape[0]),
#                idxs_max_times_less_than_start
#            ]
#        deltas_at_most_recent_time = deltas[idxs_deltas].squeeze(-1)
#        shifted_end_of_window = \
#            max_times_less_than_start + time_delta + deltas_at_most_recent_time
#
#        shifted_start_of_window = \
#            max_times_less_than_start + deltas_at_most_recent_time
#
##        print(shifted_start_of_window.shape, shifted_end_of_window.shape, 'start/end shapes')
#
#
#        end_surv = torch.exp(
#            self.compute_logsurv(shifted_end_of_window, global_theta)
#        )
#
#        start_surv = torch.exp(
#            self.compute_logsurv(shifted_start_of_window, global_theta)
#        )
#        
#        ret = (1. - end_surv/start_surv)
#
#        return ret
        
################################################ For cases in [s, s + \delta s] with fixed 
################################################ integration boundaries        

    def compute_cond_prob_over_window(
        self, deltas, batch, global_theta,
        start_time, time_delta
    ):
        # first find j*

        max_times_less_than_start, deltas_at_most_recent_time = \
            self.find_most_recent_times_and_deltas(deltas, batch, start_time)

        shifted_end_of_window = \
            start_time + time_delta + deltas_at_most_recent_time

        shifted_start_of_window = \
            start_time + deltas_at_most_recent_time

#        print(shifted_start_of_window.shape, shifted_end_of_window.shape, 'start/end shapes')


        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, global_theta)
        )

        normalization = torch.exp(
            self.compute_logsurv(
                max_times_less_than_start + deltas_at_most_recent_time, 
                global_theta
            )
        )        

        ret = 1./normalization * (start_surv - end_surv)

        return ret


    def find_most_recent_times_and_deltas(self, deltas, batch, start_time):
        pad = torch.zeros(batch.cov_times.shape)
        cov_times_less_than_start = torch.where(   
            batch.cov_times < start_time,
            batch.cov_times, pad
        )
        max_times_less_than_start, idxs_max_times_less_than_start = \
            torch.max(cov_times_less_than_start, dim=1)      
 
#        print(deltas.shape, deltas.squeeze(-1)[:, idxs_max_times_less_than_start].shape, max_times_less_than_start.shape)
        idxs_deltas = \
            [
                torch.arange(0, idxs_max_times_less_than_start.shape[0]),
                idxs_max_times_less_than_start
            ]
        deltas_at_most_recent_time = deltas[idxs_deltas].squeeze(-1)
        return max_times_less_than_start, deltas_at_most_recent_time

    def compute_most_recent_CDF(
        self, deltas, batch, global_theta,
        start_time, time_delta
    ):

        # for the 'dynamic deephit version' of the c-index with integrating from
        # 0 to S + \Delta S
        max_times_less_than_start, deltas_at_most_recent_time = \
            self.find_most_recent_times_and_deltas(deltas, batch, start_time)


        shifted_end_of_window = \
            start_time + time_delta + deltas_at_most_recent_time

        #shifted_start_of_window = \
        #    start_time + deltas_at_most_recent_time

#        print(shifted_start_of_window.shape, shifted_end_of_window.shape, 'start/end shapes')


        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

#        start_surv = torch.exp(
#            self.compute_logsurv(shifted_start_of_window, global_theta)
#        )

        normalization = torch.exp(
            self.compute_logsurv(
                max_times_less_than_start + deltas_at_most_recent_time, 
                global_theta
            )
        )        

        ret = 1. - end_surv/normalization 
        return ret
        

def print_grad(grad):
    print(grad) 
