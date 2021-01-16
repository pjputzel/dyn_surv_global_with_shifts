import torch
import torch.nn as nn

def print_num_nans(x):
    print(torch.sum(torch.isnan(x) | ~torch.isfinite(x)))

class ThetaIJBaseLogProbCalculator(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    def forward(self, thetas, batch, global_theta):
        global_theta = None
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
        if self.params['avg_per_seq']:
            # prevents long sequences from
            # dominating the loss
            logprob = logprob/batch.traj_lens.unsqueeze(1)
        ret = torch.mean(logprob)
        return ret
    


    def compute_logpdf(self, shifted_event_times, thetas):
        raise NotImplementedError('Compute logpdf must be defined in subclasses of DeltaIJLogProbCalculator') 

    def compute_logsurv(self, shifted_event_times, thetas):
        raise NotImplementedError('Compute logsurv must be defined in subclasses of DeltaIJLogProbCalculator') 
    
    def compute_lognormalization(self, shifted_cov_times, thetas): 
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
            batch.get_most_recent_times_and_idxs_before_start(start_time)
 
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

    def compute_survival_probability(
        self, thetas, batch, global_theta,
        start_time, time_delta=None
    ):
        global_theta = None
        # doesn't actually use the time delta
        # since survival is from S -> infty
        
        _, thetas_at_most_recent_time = \
            self.find_most_recent_times_and_thetas(thetas, batch, start_time)
        shifted_start_times = start_time + thetas_at_most_recent_time        

        survival_prob = torch.exp( 
            self.compute_logsurv(
                shifted_start_times,
                thetas_at_most_recent_time
            )
        )
        return survival_prob

    def compute_cond_prob_from_start_to_event_time(
        self, thetas, batch, global_theta,
        start_time, time_delta='to_true_event_time'
    ):
        global_theta = None
        max_times_less_than_start, thetas_at_most_recent_time = \
            self.find_most_recent_times_and_thetas(thetas, batch, start_time)
        end_time = batch.event_times
        
        start_surv = torch.exp(
            self.compute_logsurv(start_time, thetas_at_most_recent_time)
        )

        end_surv = torch.exp(
            self.compute_logsurv(end_time, thetas_at_most_recent_time)
        )

        normalization = torch.exp(
            self.compute_logsurv(
                max_times_less_than_start, 
                thetas_at_most_recent_time
            )
        )

        ret = (1/normalization) * (start_surv - end_surv)
        # use -1 as filler for risks that shouldn't be used
        # in C-index from start calculation 
        ret = torch.where(
            batch.event_times >= start_time,
            ret, -1 * torch.ones(ret.shape)
        )
        return ret

    def compute_cond_prob_from_start_to_event_time_ik(
        self, thetas, batch, global_theta,
        start_time, event_time_i, k 
    ):

        global_theta = None

        max_times_less_than_start, thetas_at_most_recent_time = \
            self.find_most_recent_times_and_thetas(thetas, batch, start_time)
        end_time = event_time_i
        
        start_surv = torch.exp(
            self.compute_logsurv(start_time, thetas_at_most_recent_time[k])
        )

        end_surv = torch.exp(
            self.compute_logsurv(end_time, thetas_at_most_recent_time[k])
        )

        normalization = torch.exp(
            self.compute_logsurv(
                max_times_less_than_start[k], 
                thetas_at_most_recent_time
            )
        )

        ret = (1/normalization) * (start_surv - end_surv)
        return ret

    def compute_cond_probs_truncated_at_S_over_window(self,
        thetas, batch, global_theta, start_time, time_delta
    ):
        global_theta = None
        max_times_less_than_start, thetas_at_most_recent_time = \
            self.find_most_recent_times_and_thetas(thetas, batch, start_time)
   
        shifted_end_of_window = \
            start_time + time_delta + thetas_at_most_recent_time

        shifted_start_of_window = \
            start_time + thetas_at_most_recent_time

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, thetas_at_most_recent_time)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, thetas_at_most_recent_time)
        )
#        check = (end_surv - start_surv < 1e-121)
#        print(torch.sum(check)/end_surv.shape[0])
#        print(end_surv, start_surv)
        ret = 1. - end_surv/start_surv
#        print(torch.sum(torch.isnan(ret)))
        return ret

    def compute_cond_probs_truncated_at_S_to_event_times_i(self,
        thetas, batch, global_theta, start_time, time_delta
    ):
        global_theta = None
        max_times_less_than_start, thetas_at_most_recent_time = \
            self.find_most_recent_times_and_thetas(thetas, batch, start_time)
        sorted_event_times, sort_idxs = torch.sort(batch.event_times)
        sorted_thetas = thetas_at_most_recent_time[sort_idxs]
   
        shifted_end_of_window = \
            sorted_event_times.reshape(-1, 1) + sorted_thetas.reshape(1, -1)

        shifted_start_of_window = \
            start_time + sorted_thetas

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, sorted_thetas)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, sorted_thetas)
        )

        ret = 1. - end_surv/start_surv
        unc_at_risk = \
            ~batch.censoring_indicators[sort_idxs].bool() &\
            (batch.event_times[sort_idxs] > start_time)
        ret = ret[unc_at_risk]
        return ret
def print_grad(grad):
    print(grad) 
