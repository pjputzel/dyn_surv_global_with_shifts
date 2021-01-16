import torch
import torch.nn as nn

def print_num_nans(x):
    print(torch.sum(torch.isnan(x) | ~torch.isfinite(x)))

class GlobalParamBaseLogProbCalculator(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    def forward(self, deltas, batch, global_theta):
        logpdf = self.compute_logpdf(batch.event_times, global_theta)
        logsurv = self.compute_logsurv(batch.event_times, global_theta)

        logprob = torch.where(
            batch.censoring_indicators.unsqueeze(1).bool(),
            logsurv, logpdf
        )

        ret = torch.mean(logprob)
        return ret
    

    def compute_logpdf(self, shifted_event_times, global_theta):
        raise NotImplementedError('Compute logpdf must be defined in subclasses of DeltaIJLogProbCalculator') 

    def compute_logsurv(self, shifted_event_times, global_theta):
        raise NotImplementedError('Compute logsurv must be defined in subclasses of DeltaIJLogProbCalculator') 
    
    def compute_lognormalization(self, shifted_cov_times, global_theta): 
        raise NotImplementedError('Compute lognormalization must be defined in subclasses of DeltaIJLogProbCalculator') 

