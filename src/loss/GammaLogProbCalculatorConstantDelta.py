import torch
import torch.nn as nn
import numpy as np


def estimate_lower_reg_incomplete_gamma_with_series(gamma_concentration, x_boundary, n_terms=20):
    x_boundary_lengths = [x[~(x == 0)].shape[0] for x in x_boundary]
    if len(x_boundary.shape) > 1:
        prefactor = torch.zeros(x_boundary.shape[0], np.max(x_boundary_lengths))
        for i, length in enumerate(x_boundary_lengths):
            prefactor[i, :length] = x_boundary[i, :length] ** gamma_concentration[i] * torch.exp(-x_boundary[i, :length])
    else:
        prefactor = x_boundary**gamma_concentration * torch.exp(-x_boundary) #* torch.as_tensor(~(x_boundary == 0), dtype=torch.double)
    sum_of_terms = 0
    for i in range(n_terms):
        denominator = gamma_concentration
        for j in range(i):
            denominator = denominator * (gamma_concentration + j + 1)
            
        term_i = x_boundary**i / denominator
        sum_of_terms = sum_of_terms + term_i

    return prefactor * sum_of_terms/torch.exp(torch.lgamma(gamma_concentration)) 


class GammaLogProbCalculatorConstantDelta(nn.Module):
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
        shifted_cov_times = torch.max(batch.cov_times, dim=1)[0] + deltas.squeeze(1)
        #deltas.register_hook(print_grad)
        #print(deltas, 'deltas')
        #print(torch.max(batch.cov_times, dim=1)[0])
        #print(shifted_cov_times, 'shifted cov time')
        return shifted_event_times, shifted_cov_times



    def compute_logpdf(self, shifted_event_times, global_theta):
        print(global_theta)
        alpha = global_theta[0]
        beta = global_theta[1]
        log_pdf = \
            alpha * torch.log(beta) - torch.lgamma(alpha) + \
            (alpha - 1) * torch.log(shifted_event_times) - \
            beta * shifted_event_times

        return log_pdf
        

    def compute_logsurv(self, shifted_event_times, global_theta):
        alpha = global_theta[0]
        beta = global_theta[1]
        gamma_cdf = estimate_lower_reg_incomplete_gamma_with_series(
            alpha, shifted_event_times * beta
        )
        
        
        return 1. - gamma_cdf
    
    def compute_lognormalization(self, shifted_cov_times, global_theta):
        return self.compute_logsurv(shifted_cov_times, global_theta)




