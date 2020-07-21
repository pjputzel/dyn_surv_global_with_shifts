import torch
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1]), 'weibull':(2, [1, 1]), 'rayleigh':(1, [1])}


def print_nans(tensor):
    print(tensor[torch.isnan(tensor)])

class DummyGlobalModel(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type


        self.global_param_logspace = nn.Parameter(torch.rand(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
        self.deltas_fixed_to_zero = False
        
    def forward(self, batch):
        batch_cov_times = batch.cov_times
        # dummy model to evaluate the global model alone
        pred_deltas = torch.zeros(batch_cov_times.shape[0], batch_cov_times.shape[1], 1)
        hidden_states = torch.zeros(batch_cov_times.shape[0])
        next_step_cov_preds = torch.tensor(batch_cov_times.shape[0])
        #print(self.linear.weight, self.linear.bias)
        return pred_deltas, hidden_states, next_step_cov_preds

    
    def get_global_param(self):
        return torch.exp(-self.global_param_logspace)

    def freeze_global_param(self):
        self.global_param_logspace.requires_grad = False

    def fix_deltas_to_zero(self):
        self.deltas_fixed_to_zero = True

    def unfix_deltas_to_zero(self):
        self.deltas_fixed_to_zero = False

