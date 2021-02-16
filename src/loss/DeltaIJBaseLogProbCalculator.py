import torch
import torch.nn as nn

def print_num_nans(x):
    print(torch.sum(torch.isnan(x) | ~torch.isfinite(x)))

class DeltaIJBaseLogProbCalculator(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    def forward(self, deltas, batch, global_theta):
        #print(torch.sum(torch.isnan(batch.censoring_indicators)))
        #print(torch.sum(torch.isnan(batch.event_times)))
#        deltas.register_hook(print_grad)        
        shifted_event_times, shifted_cov_times = self.compute_shifted_times(deltas, batch)
        logpdf = self.compute_logpdf(shifted_event_times, global_theta)
        logsurv = self.compute_logsurv(shifted_event_times, global_theta)
        lognormalization = self.compute_lognormalization(shifted_cov_times, global_theta)

        logprob = torch.where(
            batch.censoring_indicators.unsqueeze(1).bool(),
            logsurv, logpdf
        )
        logprob = logprob - lognormalization

        # zero out the padded values
        padding_indicators = \
            (batch.cov_times == 0) #&\
            #torch.cat([torch.zeros(batch.cov_times.shape[0], 1), torch.ones(batch.cov_times.shape[0], batch.cov_times.shape[1] - 1)], dim=1).bool()
        # first time is never padding despite being 0
        padding_indicators[0, :] = False

#        logprob = torch.where(\
#            padding_indicators,
#            torch.zeros(logprob.shape), logprob
#        )
        logprob = (~padding_indicators) * logprob
        if self.params['avg_per_seq']:
            # prevents long sequences from dominating
            # the loss
            logprob = logprob/batch.traj_lens.unsqueeze(1)
        ret = torch.mean(logprob)
        return ret
    
    def compute_shifted_times(self, deltas, batch):
        shifted_event_times = batch.event_times.unsqueeze(1) + deltas.squeeze(2)
        idxs = (shifted_event_times == 0).nonzero()
#        print((shifted_event_times == 0).nonzero())
#        print(batch.cov_times[idxs[:, 0], idxs[:, 1]])
#        print([len(b[~(b == 0)]) for b in batch.cov_times[idxs[:, 0]]])
#        print(batch.event_times[idxs[:, 0]])
#        print(batch.censoring_indicators[idxs[:, 0]])
        
        shifted_cov_times = batch.cov_times + deltas.squeeze(2)
#        print(shifted_cov_times[0:5, 0:30])
#        print(torch.sum(~(shifted_cov_times == 0)))
        return shifted_event_times, shifted_cov_times


    def compute_logpdf(self, shifted_event_times, global_theta):
        raise NotImplementedError('Compute logpdf must be defined in subclasses of DeltaIJLogProbCalculator') 

    def compute_logsurv(self, shifted_event_times, global_theta):
        raise NotImplementedError('Compute logsurv must be defined in subclasses of DeltaIJLogProbCalculator') 
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        raise NotImplementedError('Compute lognormalization must be defined in subclasses of DeltaIJLogProbCalculator') 

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
        max_times_less_than_start, idxs_max_times_less_than_start = \
            batch.get_most_recent_times_and_idxs_before_start(start_time)
 
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

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        normalization = torch.exp(
            self.compute_logsurv(
                max_times_less_than_start + deltas_at_most_recent_time, 
                global_theta
            )
        )        

        ret = 1. - end_surv/normalization 
        return ret

    def compute_survival_probability(
        self, deltas, batch, global_theta,
        start_time, time_delta=None
    ):
        # doesn't actually use the time delta
        # since survival is from S -> infty
        
        _, deltas_at_most_recent_time = \
            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
        shifted_start_times = start_time + deltas_at_most_recent_time        

        survival_prob = torch.exp( 
            self.compute_logsurv(
                shifted_start_times,
                global_theta
            )
        )
        return survival_prob

#    def compute_cond_prob_from_start_to_event_time(
#        self, deltas, batch, global_theta,
#        start_time, time_delta='to_true_event_time'
#    ):
#
#        max_times_less_than_start, deltas_at_most_recent_time = \
#            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
#        shifted_start_times = start_time + deltas_at_most_recent_time        
#        shifted_event_times = batch.event_times + deltas_at_most_recent_time
#        
#        start_surv = torch.exp(
#            self.compute_logsurv(shifted_start_times, global_theta)
#        )
#
#        end_surv = torch.exp(
#            self.compute_logsurv(shifted_event_times, global_theta)
#        )
#
#        normalization = torch.exp(
#            self.compute_logsurv(
#                max_times_less_than_start + deltas_at_most_recent_time, 
#                global_theta
#            )
#        )
#
#        ret = (1/normalization) * (start_surv - end_surv)
#        # use -1 as filler for risks that shouldn't be used
#        # in C-index from start calculation 
#        ret = torch.where(
#            batch.event_times >= start_time,
#            ret, -1 * torch.ones(ret.shape)
#        )
#        return ret
#
#    def compute_cond_prob_from_start_to_event_time_ik(
#        self, deltas, batch, global_theta,
#        start_time, event_time_i, k 
#    ):
#
#        max_times_less_than_start, deltas_at_most_recent_time = \
#            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
#        shifted_event_time = event_time_i + deltas_at_most_recent_time[k]
#        shifted_start_time = start_time + deltas_at_most_recent_time[k]
#        
#        start_surv = torch.exp(
#            self.compute_logsurv(shifted_start_time, global_theta)
#        )
#
#        end_surv = torch.exp(
#            self.compute_logsurv(shifted_event_time, global_theta)
#        )
#
#        normalization = torch.exp(
#            self.compute_logsurv(
#                max_times_less_than_start[k] + deltas_at_most_recent_time[k], 
#                global_theta
#            )
#        )
#
#        ret = (1/normalization) * (start_surv - end_surv)
#        return ret

    '''
    returns in sorted order (by event times) the conditional probability over the region
    t_kj* -> T_i for use in computing time dependent c-index from most recent time version
    return is of size N X N (before when buggy was U X N)
    '''

    def compute_cond_probs_k_from_start_to_event_times_i(self, 
        deltas, batch, global_theta,
        start_time, time_delta 
    ):
        # time delta not used here, included to make code cleaner and avoid extra
        # if/else
        time_delta = None        
        max_times_less_than_start, deltas_at_most_recent_time = \
            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
        sorted_event_times, sort_idxs = torch.sort(batch.event_times)
        sorted_deltas = deltas_at_most_recent_time[sort_idxs]
        sorted_tijs = max_times_less_than_start[sort_idxs]
   
        shifted_end_of_window = \
            sorted_event_times.reshape(-1, 1) + sorted_deltas.reshape(1, -1)

        shifted_start_of_window = \
            start_time + sorted_deltas

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, global_theta)
        )

        normalization = torch.exp(
            self.compute_logsurv(
                sorted_tijs + \
                sorted_deltas,
                global_theta
            )
        )        

        ret = 1./normalization * (start_surv - end_surv)
#        unc_at_risk = \
#            ~batch.censoring_indicators[sort_idxs].bool() &\
#            (batch.event_times[sort_idxs] > start_time)
#        ret = ret[unc_at_risk]
        return ret

    def compute_cond_probs_truncated_at_S_over_window(self,
        deltas, batch, global_theta, start_time, time_delta
    ):
        max_times_less_than_start, deltas_at_most_recent_time = \
            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
   
        shifted_end_of_window = \
            start_time + time_delta + deltas_at_most_recent_time

        shifted_start_of_window = \
            start_time + deltas_at_most_recent_time

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, global_theta)
        )
#        check = (end_surv - start_surv < 1e-121)
#        print(torch.sum(check)/end_surv.shape[0])
#        print(end_surv, start_surv)
        ret = 1. - end_surv/start_surv
#        print(torch.sum(torch.isnan(ret)))
        return ret

    def compute_cond_probs_truncated_at_S_to_event_times_i(self,
        deltas, batch, global_theta, start_time, time_delta
    ):
        max_times_less_than_start, deltas_at_most_recent_time = \
            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
        sorted_event_times, sort_idxs = torch.sort(batch.event_times)
        sorted_deltas = deltas_at_most_recent_time[sort_idxs]
   
        shifted_end_of_window = \
            sorted_event_times.reshape(-1, 1) + sorted_deltas.reshape(1, -1)

        shifted_start_of_window = \
            start_time + sorted_deltas

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, global_theta)
        )

        ret = 1. - end_surv/start_surv
#        unc_at_risk = \
#            ~batch.censoring_indicators[sort_idxs].bool() &\
#            (batch.event_times[sort_idxs] > start_time)
#        ret = ret[unc_at_risk]
        return ret

    def compute_cond_probs_from_cov_time_k_to_event_times_i(self,
        deltas, batch, global_theta, start_time, time_delta
    ):
        # not used for this version of the c_index
        time_delta = None
        max_times_less_than_start, deltas_at_most_recent_time = \
            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
        sorted_event_times, sort_idxs = torch.sort(batch.event_times)
        sorted_deltas = deltas_at_most_recent_time[sort_idxs]
        sorted_tijs = max_times_less_than_start[sort_idxs]
   
        shifted_end_of_window = \
            sorted_event_times.reshape(-1, 1) + sorted_deltas.reshape(1, -1)
    
        shifted_start_of_window = \
            sorted_tijs + sorted_deltas 

        end_surv = torch.exp(
            self.compute_logsurv(shifted_end_of_window, global_theta)
        )

        start_surv = torch.exp(
            self.compute_logsurv(shifted_start_of_window, global_theta)
        )


        ret = 1. - (end_surv/start_surv)
        unc_at_risk = \
            ~batch.censoring_indicators[sort_idxs].bool() &\
            (batch.event_times[sort_idxs] > start_time)
        ret = ret[unc_at_risk]
        return ret
        


#    def compute_cond_probs_k_from_start_to_event_times_i(self,
#        pred_params, data, global_param,
#        start_time, time_delta 
#    ):
#        sorted_event_times, sort_idxs = torch.sort(data.event_times)
#        uncensored_sorted_times = sorted_event_times[
#            ~data.censoring_indicators[sort_idxs].bool()
#        ]
#        n_uncensored = torch.sum(data.censoring_indicators == 0)
#        
#        risks_ik = -1 * torch.ones(n_uncensored, len(data.event_times))
#        iterate_over = enumerate(zip(sort_idxs, uncensored_sorted_times))
#        for idx_unc, (idx_i, time_i) in iterate_over:
#            risks_ik[idx_unc, idx_i + 1:] = \
#                self.compute_cond_probs_k_from_start_to_time_i(
#                    self, data, idx_i, 
#                    global_param, start_time
#                )
#                
#            
#    def compute_cond_probs_k_from_start_to_time_i(self,
#        data, global_param, idx_i, start_time
#    ):
#        
#        
#        # first find j*
#
#        max_times_less_than_start, deltas_at_most_recent_time = \
#            self.find_most_recent_times_and_deltas(deltas, batch, start_time)
#
#        shifted_end_of_window = \
#            batch.event_times[idx_i] + deltas_at_most_recent_time[idx_i + 1:]
#
#        shifted_start_of_window = \
#            start_time + deltas_at_most_recent_time[idx_i + 1:]
#
#        end_surv = torch.exp(
#            self.compute_logsurv(shifted_end_of_window, global_theta)
#        )
#
#        start_surv = torch.exp(
#            self.compute_logsurv(shifted_start_of_window, global_theta)
#        )
#
#        normalization = torch.exp(
#            self.compute_logsurv(
#                max_times_less_than_start[idx_i + 1:] + \
#                deltas_at_most_recent_time[idx_i + 1:],
#                global_theta
#            )
#        )        
#
#        ret = 1./normalization * (start_surv - end_surv)
#

def print_grad(grad):
    print(torch.isnan(grad).nonzero())
