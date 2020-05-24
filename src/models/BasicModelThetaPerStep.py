import torch
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1])}


class BasicModelThetaPerStep(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type
        # plus one is for the timestamp -> needs to be updated to 2 * covariate_dim + 1 to account for missing indicators
        # TODO add dropout back in!!
        self.RNN = nn.GRU(\
            self.params['covariate_dim'] + 1, self.params['hidden_dim'],
            #dropout=self.params['dropout']
        )

        self.params_fc_layer = nn.Linear(
            self.params['hidden_dim'],
            SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]
        )

        # TODO: this should really take in the last hidden state
        # and the time of the next measurement and then predict
        # the next measurement. 'given state and time what's the
        # next measurement'. Will likely have to change in the
        # make cov preds function
        self.cov_fc_layer = nn.Linear(\
            self.params['hidden_dim'] + 1,
            self.params['covariate_dim']
            #self.params['covariate_dim'] + 1 to predict time too
        )
        
    def forward(self, packed_sequence_batch):
        batch_size = packed_sequence_batch.batch_sizes[0]

        h_0 = Variable(torch.randn(\
            1, batch_size, 
            self.params['hidden_dim']
        ))

        # eventually may add attention by using the hidden_states/'output' of the GRU
        hidden_states, _ = self.RNN(packed_sequence_batch, h_0)
        unpacked_hidden_states, lengths = self.unpack_and_permute(hidden_states)

        fc_output = self.params_fc_layer(unpacked_hidden_states)

        pred_params = torch.exp(-fc_output)

        batch_covs_unpacked, _ = self.unpack_and_permute(packed_sequence_batch)
        next_step_cov_preds = self.make_next_step_cov_preds(
            unpacked_hidden_states, 
            batch_covs_unpacked, 
            lengths
        ) 

        return pred_params, unpacked_hidden_states, next_step_cov_preds


    # TODO: this should really take in the last hidden state
    # and the time of the next measurement and then predict
    # the next measurement. 'given state and time what's the
    # next measurement'. Will likely have to change in the
    # make cov preds function
    def make_next_step_cov_preds(self, unpacked_hidden_states, batch_covs_unpacked, lengths): 
        unpacked_hidden_states_with_times = torch.cat(
            [unpacked_hidden_states, batch_covs_unpacked[:, :, 0].unsqueeze(2)], 
            axis=2
        )

        next_step_cov_preds = torch.zeros(
            unpacked_hidden_states.shape[1] - 1,
            batch_covs_unpacked.shape[0],
            self.params['covariate_dim'] + 1
        )

        iterations = enumerate(zip(unpacked_hidden_states_with_times, lengths))        
        for batch, (hidden_states_per_step, length) in iterations:
            for i in range(length - 1):
                if self.params['hidden_dim'] == 1:
                    fc_arg = hidden_states_per_step[i].reshape(1, 1)
                next_step_cov_preds[i, batch, :] = self.cov_fc_layer(hidden_states_per_step[i])
        next_step_cov_preds = next_step_cov_preds.permute(1, 0, 2)

        return next_step_cov_preds
    
    def freeze_rnn_parameters(self):
        for param in self.RNN.parameters():
            param.requires_grad = False

    def freeze_cov_pred_parameters(self):
        for param in self.cov_fc_layer.parameters():
            param.requires_grad = False

    def unpack_and_permute(self, packed_tensor):
        unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(packed_tensor)
        unpacked = unpacked.permute(1, 0, 2)
        return unpacked, lens
    
    def get_global_param(self):
        return torch.exp(-self.params_fc_layer.bias)
