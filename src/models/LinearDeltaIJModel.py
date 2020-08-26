import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1]), 'weibull':(2, [1, 1]), 'rayleigh':(1, [1]), 'chen2000': (2, [1, 1]), 'emwe':(4, [1,1,1,1])}


def print_nans(tensor):
    print(tensor[torch.isnan(tensor)])

class LinearDeltaIJModel(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type

        # eventually multiply dynamic cov dim by two for the missing indicators
        # no plus one since not using time here since with time its giving weird solutions
        # (just using negative of the time it seems)
        self.linear = nn.Linear(
            int(2 * self.params['dynamic_cov_dim'] + self.params['static_cov_dim'] + 1),
            1
        )
        # weights for the linear layer are set to around one in order to
        # get a random initialization with most deltas around zero
#        self.linear.weight.data.normal_(0., 1./(np.sqrt(self.linear.in_features)))
#        self.linear.bias.data.normal_(0., 1./(np.sqrt(self.linear.in_features)))
#        self.linear.bias.data.fill_(1., 2.)

        self.global_param_logspace = nn.Parameter(torch.rand(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
#        self.global_param_logspace = nn.Parameter(torch.zeros(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
        self.deltas_fixed_to_zero = False
        
    def forward(self, batch):
        batch_covs = batch.get_unpacked_padded_cov_trajs()

        static_covs = batch.static_covs
        static_covs[torch.isnan(static_covs)] = -1
#        print(batch.cov_times.shape, batch_covs.shape, static_covs.shape)
        all_data = torch.cat([batch_covs, static_covs.unsqueeze(1).repeat(1, batch_covs.shape[1], 1)], dim=2)
#        print(batch.cov_times.shape, all_data.shape)
        if not self.deltas_fixed_to_zero:
            softplus = torch.nn.functional.softplus(self.linear(all_data))
#            pred_deltas = torch.exp(-self.linear(all_data)) - batch.cov_times.unsqueeze(-1)
            pred_deltas = softplus - batch.cov_times.unsqueeze(-1)
        else:
            pred_deltas = torch.zeros(all_data.shape[0], 1)
#        print(pred_deltas.shape)
        # this model has no hidden states
        # this model has no next step cov preds
        # so the last two outputs are just zeros
        hidden_states = torch.zeros(batch_covs.shape[0])
        next_step_cov_preds = torch.tensor(batch_covs.shape[0])
#        print(pred_deltas)
#        print(self.linear.weight, self.linear.bias, torch.exp(-self.global_param_logspace))
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

