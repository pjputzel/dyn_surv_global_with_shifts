
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1]), 'weibull':(2, [1, 1]), 'rayleigh':(1, [1]), 'chen2000': (2, [1, 1]), 'emwe':(4, [1,1,1,1]), 'gompertz': (2, [1, 1])}


def print_nans(tensor):
    print(tensor[torch.isnan(tensor)])

class LinearDeltaIJModelNumVisitsOnly(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type

        # eventually multiply dynamic cov dim by two for the missing indicators
        # no plus one since not using time here since with time its giving weird solutions
        # (just using negative of the time it seems)
        self.linear = nn.Linear(
            1, #just using the number of events as a predictor
            1
        )
        # weights for the linear layer are set to around one in order to
        # get a random initialization with most deltas around zero
#        self.linear.weight.data.normal_(0., 1./(np.sqrt(self.linear.in_features)))
#        self.linear.bias.data.normal_(0., 1./(np.sqrt(self.linear.in_features)))
#        self.linear.bias.data.fill_(1., 2.)
        print(distribution_type)
        self.global_param_logspace = nn.Parameter(torch.rand(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
#        self.global_param_logspace = nn.Parameter(torch.zeros(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
        self.deltas_fixed_to_zero = False
        
    def forward(self, batch):
        all_data = batch.cov_times.unsqueeze(-1)
#        print(batch.cov_times.shape, all_data.shape)
        if not self.deltas_fixed_to_zero:
            softplus = torch.nn.functional.softplus(self.linear(all_data), beta=100)
            print(softplus)
            pred_deltas = softplus - batch.cov_times.unsqueeze(-1)
#            pred_deltas = torch.exp(-self.linear(all_data)) - batch.cov_times.unsqueeze(-1)
        else:
            pred_deltas = torch.zeros(all_data.shape[0], 1)
#        print(pred_deltas.shape)
        # this model has no hidden states
        # this model has no next step cov preds
        # so the last two outputs are just zeros
        hidden_states = torch.zeros(all_data.shape[0])
        next_step_cov_preds = torch.tensor(all_data.shape[0])
#        print(pred_deltas[0], all_data[0, :, 0:2])
#        print(self.linear.weight[0][0:2], self.linear.bias, torch.exp(-self.global_param_logspace))
#        print(all_data[0:5])
        return pred_deltas, hidden_states, next_step_cov_preds

    
    def get_global_param(self):
#        if self.distribution_type == 'weibull':
#            scale = torch.exp(-self.global_param_logspace[0])
            # clamp k at two in order to avoid infinite gradients near zero
#            shape = torch.exp(-self.global_param_logspace[1]) + 2.
#            return torch.cat([scale.unsqueeze(0), shape.unsqueeze(0)])
        return torch.exp(-self.global_param_logspace)

    def freeze_global_param(self):
        self.global_param_logspace.requires_grad = False

    def fix_deltas_to_zero(self):
        self.deltas_fixed_to_zero = True

    def unfix_deltas_to_zero(self):
        self.deltas_fixed_to_zero = False

