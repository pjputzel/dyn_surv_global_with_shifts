import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1]), 'weibull':(2, [1, 1]), 'rayleigh':(1, [1]), 'gompertz':(2, [1,1]), 'chen2000':(2, [1, 1]), 'folded_normal':(2, [1, 1]), 'log_normal': (2, [0, 1])}


def print_nans(tensor):
    print(tensor[torch.isnan(tensor)])

class DummyGlobalModel(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type


        if self.params['param_init_scales']:
            num_params = SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]
            assert len(self.params['param_init_scales']) == num_params
            params = []
            for param in range(num_params):
                params.append(-np.log(torch.rand(1) * self.params['param_init_scales'][param]))
            self.global_param_logspace = nn.Parameter(torch.tensor(params, dtype=torch.float32))
#            print(self.global_param_logspace, torch.exp(-self.global_param_logspace))
        else:
            self.global_param_logspace = nn.Parameter(torch.rand(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
#        self.global_param_logspace = nn.Parameter(torch.rand(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
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
    
    def set_global_param(self, global_param):
        with torch.no_grad():
            self.global_param_logspace.copy_(
                nn.Parameter(-torch.log(global_param))
            )

    def freeze_global_param(self):
        self.global_param_logspace.requires_grad = False

    def fix_deltas_to_zero(self):
        self.deltas_fixed_to_zero = True

    def unfix_deltas_to_zero(self):
        self.deltas_fixed_to_zero = False

