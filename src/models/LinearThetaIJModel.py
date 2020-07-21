import torch
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1]), 'weibull':(2, [1, 1]), 'rayleigh':(1, [1]), 'chen2000': (2, [1, 1]), 'emwe':(4, [1,1,1,1])}


def print_nans(tensor):
    print(tensor[torch.isnan(tensor)])

class LinearThetaIJModel(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type

        # eventually multiply dynamic cov dim by two for the missing indicators
        self.linear = nn.Linear(
            int(2 * self.params['dynamic_cov_dim'] + self.params['static_cov_dim'] + 1),
            SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]
        )

        
    def forward(self, batch):
        batch_covs = batch.get_unpacked_padded_cov_trajs()

        static_covs = batch.static_covs
        static_covs[torch.isnan(static_covs)] = -1
#        print(batch.cov_times.shape, batch_covs.shape, static_covs.shape)
        all_data = torch.cat([batch_covs, static_covs.unsqueeze(1).repeat(1, batch_covs.shape[1], 1)], dim=2)
#        print(batch.cov_times.shape, all_data.shape)
        pred_thetas = torch.exp(-self.linear(all_data))
#        print(pred_deltas.shape)
        # this model has no hidden states
        # this model has no next step cov preds
        # so the last two outputs are just zeros
        hidden_states = torch.zeros(batch_covs.shape[0])
        next_step_cov_preds = torch.tensor(batch_covs.shape[0])
        #print(self.linear.weight, self.linear.bias)
        return pred_thetas, hidden_states, next_step_cov_preds

    
    def get_global_param(self):
        return torch.exp(-self.linear.bias) 



