import torch
import torch.nn as nn

class ExponentialLogProbCalculator(nn.Module):
    
    def __init__(self, logprob_params):
        super().__init__()
        self.params = logprob_params

    # if pred params is a single theta make sure to 
    # unsqueeze the third dimension before calling here!
    # this should really occur in model forwards call so update
    # that if you run into issues here.
    def forward(self, pred_params, batch):
        # exp only has a single parameter
        pred_params = pred_params.squeeze(2)
        shifted_times = self.compute_shifted_times(batch)
        logpdf = self.compute_logpdf(pred_params, shifted_times)
        logsurv = self.compute_logsurv(pred_params, shifted_times)
       
        logprob_per_i = torch.where(
            batch.censoring_indicators.unsqueeze(1).bool(),
            logsurv, logpdf
        )

        if self.params['avg_per_seq']:
            avg_per_i = 1/(batch.trajectory_lengths) * torch.sum(logprob_per_i, dim=1)
        else:
            message = 'Averaging type %s not recognized' %s
            raise ValueError(message) 

        return torch.mean(avg_per_i)

    
    def compute_shifted_times(self, batch):
        pad_placholder = torch.zeros(batch.cov_times.shape)
        shifted_non_pad = batch.event_times.unsqueeze(1) - batch.cov_times
        # first event times are zero even though not padded
        is_not_first_event = torch.cat(
            [
                torch.zeros(batch.cov_times.shape[0], 1), 
                torch.ones(batch.cov_times.shape[0], batch.cov_times.shape[1] - 1)
            ],
            dim=1
        )
        is_padding = (batch.cov_times == 0) & (is_not_first_event.bool())
        ret = torch.where(is_padding, pad_placholder, shifted_non_pad)
        return ret


    def compute_logpdf(self, pred_params, shifted_times):
        logpdf = torch.log(pred_params) - pred_params * shifted_times
        pad_placholder = torch.zeros(logpdf.shape)
        is_padding = shifted_times == 0 
        return torch.where(is_padding, pad_placholder, logpdf)
        
       
    def compute_logsurv(self, pred_params, shifted_times):    
        logsurv = -pred_params * shifted_times
        pad_placholder = torch.zeros(logsurv.shape)
        is_padding = shifted_times == 0 
        return torch.where(is_padding, pad_placholder, logsurv) 
