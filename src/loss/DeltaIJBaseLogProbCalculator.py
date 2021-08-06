import torch
import torch.nn as nn

def print_num_nans(x):
    print(torch.sum(torch.isnan(x) | ~torch.isfinite(x)))

def print_nan_idxs_in_timestep(x):
    time = 8
    print('idxs with nans:')
    idxs = torch.isnan(x)[:, time, 0] | ~torch.isfinite(x)[:, time, 0]
    print(torch.arange(x.shape[0])[idxs])
    print('grads with nans at time index %d' %time)
    print(x[idxs])

class DeltaIJBaseLogProbCalculator(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params
        # for debugging
        self.extreme_loss_counts = {}

    def forward(self, deltas, batch, global_theta):
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
        # first time is never padding despite being 0
        padding_indicators[0, :] = False

        logprob = (~padding_indicators) * logprob
        if self.params['avg_per_seq']:
            # prevents long sequences from dominating
            # the loss
            logprob = logprob/batch.traj_lens.unsqueeze(1)
        ret = torch.sum(logprob, dim=1)
        
        # for debugging individuals with extremely large loss
        nll_mean, nll_std = torch.mean(ret), torch.std(ret)
        std_thresh = 5
        idxs_in_full_data = batch.shuffled_idxs[((ret - nll_mean)**(2))**(1/2) > std_thresh * nll_std]
        for idx in idxs_in_full_data:
            idx = int(idx.cpu().detach().numpy())
            if not idx in list(self.extreme_loss_counts.keys()):
                self.extreme_loss_counts[idx] = 0
            self.extreme_loss_counts[idx] = self.extreme_loss_counts[idx] + 1

        ret = ret.mean()
        return ret
    
    def compute_shifted_times(self, deltas, batch):
        shifted_event_times = batch.event_times.unsqueeze(1) + deltas.squeeze(2)
        idxs = (shifted_event_times == 0).nonzero()
        
        shifted_cov_times = batch.cov_times + deltas.squeeze(2)
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
        ret = 1. - end_surv/start_surv
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
        



def print_grad(grad):
    print(grad)
