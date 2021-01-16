
import torch
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1]), 'weibull':(2, [1, 1]), 'rayleigh':(1, [1])}


def print_nans(tensor):
    print(tensor[torch.isnan(tensor)])

class EmbeddingConstantDeltaModelLinearRegression(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type

        total_cov_dim = int(2 * self.params['dynamic_cov_dim'] + self.params['static_cov_dim'])
        print(total_cov_dim)
        self.embedding = nn.Sequential(
            torch.nn.Linear(total_cov_dim, self.params['embed_hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.params['embed_hidden_dim'], self.params['embed_output_dim']),
            torch.nn.ReLU()
        ) 


        # eventually multiply dynamic cov dim by two for the missing indicators
        # no plus one since not using time here, instead using averages
        #self.linear = nn.Linear(
        #    int(2 * self.params['dynamic_cov_dim'] + self.params['static_cov_dim']),
        #    1
        #)
        
        self.linear = nn.Linear(
            self.params['embed_output_dim'],
            1
        )

        self.global_param_logspace = nn.Parameter(torch.rand(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
        self.deltas_fixed_to_zero = False
        
    def forward(self, batch):
        batch_covs = batch.get_unpacked_padded_cov_trajs()
#########For Averaging
#        avg_covs = torch.sum(batch_covs[:, :, 1:], dim=1)/(torch.sum(1. - batch.missing_indicators, dim=1))
#        # handle cases where a person doesn't have the test at all and the denominator is zero
#        avg_covs[torch.isnan(avg_covs)] = 0 
#        missing_freq = torch.mean(batch.missing_indicators, dim=1)
#        #batch_covs_with_missingness = torch.cat([batch_covs, batch.missing_indicators], dim=2)
#        #print(avg_covs.shape, missing_freq.shape)
#        avg_batch_covs_with_missingness = torch.cat([avg_covs, missing_freq], dim=1)
#        
#        #print(batch_covs_with_missingness.shape)
#        #avg_covs = torch.mean(batch_covs_with_missingness, dim=1)
#        #print(torch.isnan(batch.static_covs))
#        static_covs = batch.static_covs
#        static_covs[torch.isnan(static_covs)] = -1
#        all_data = torch.cat([avg_batch_covs_with_missingness, static_covs], dim=1) #dim=2 before
#        print_nans(avg_batch_covs_with_missingness)
#        print_nans(batch.static_covs)
#########

########For Baseline only
        # first entry of batch cov trajs is the time stamp
        print(batch_covs.shape)
        print(batch_covs[0, :, 0])
        baseline_covs = batch_covs[:, 0, 1:]
        print(baseline_covs.shape)
        #missing_indicators = batch.missing_indicators[:, 0]
        # don't need times since they are all just at baseline t = 0
        #baseline_covs_with_missing = torch.cat([baseline_covs[:, 1:], missing_indicators], dim=1)
        static_covs = batch.static_covs
        static_covs[torch.isnan(static_covs)] = -1
        all_data = torch.cat([baseline_covs, static_covs], dim=1)
########
        #print(all_data.shape)
        # prevent the deltas from pushing the effective age negative
        # so clip the delta at the last cov event time
        # since for static model thats where we condition on

#        print(torch.max(batch.cov_times, dim=1)[0].unsqueeze(1))
        #print(all_data.shape)
        if not self.deltas_fixed_to_zero:
            #### For averaging:
            #pred_deltas = torch.exp(-self.linear(all_data)) - torch.max(batch.cov_times, dim=1)[0].unsqueeze(1)
            #### For baseline only:
            # try with and without the normalization by global param
            print(all_data.shape)
            embedded_data = self.embedding(all_data)
            pred_deltas = torch.exp(-self.linear(embedded_data))

            
        else:
            pred_deltas = torch.zeros(all_data.shape[0], 1)
        # this model has no hidden states
        # this model has no next step cov preds
        # so the last two outputs are just zeros
        hidden_states = torch.zeros(batch_covs.shape[0])
        next_step_cov_preds = torch.tensor(batch_covs.shape[0])
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

