import torch
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exponential': (1, [1]), 'lnormal':(2, [1, 1]), 'weibull':(2, [1, 1]), 'rayleigh':(1, [1]), 'chen2000':(2, [1, 1]), 'emwe':(4, [1,1,1,1]), 'gompertz':(2, [1, 1])}



class RNNDeltaIJLinearTransformModel(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type
        self.embedding = nn.Sequential(
            nn.Linear(2 * self.params['dynamic_cov_dim'] + 1, self.params['embed_hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.params['embed_hidden_dim'], self.params['embed_output_dim'])
        )
            
        # plus one is for the timestamp -> needs to be updated to 2 * covariate_dim + 1 to account for missing indicators
        # TODO add dropout back in!!
        self.RNN = nn.GRU(
            self.params['embed_output_dim'], self.params['hidden_dim'],
            dropout=self.params['dropout']
        )

        self.params_fc_layer = nn.Linear(
            self.params['hidden_dim'] + self.params['static_cov_dim'],
            1
        )

        # TODO: this should really take in the last hidden state
        # and the time of the next measurement and then predict
        # the next measurement. 'given state and time what's the
        # next measurement'. Will likely have to change in the
        # make cov preds function
        self.cov_fc_layer = nn.Linear(\
            self.params['hidden_dim'] + 1,
            self.params['dynamic_cov_dim']
            #self.params['dynamic_cov_dim'] + 1 to predict time too
        )
        self.global_param_logspace = nn.Parameter(torch.rand(SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]))
        self.init_hidden_state = nn.Parameter(
            torch.rand(1, 1, self.params['hidden_dim'])
        )

        self.deltas_fixed_to_zero = False


        
    def forward(self, batch):
        packed_sequence_batch = batch.packed_cov_trajs
        max_len = batch.max_seq_len_all_batches
        batch_covs_unpacked, _ = self.unpack_and_permute(packed_sequence_batch, max_len)
        batch_size = packed_sequence_batch.batch_sizes[0]
        
        embedded_covs = self.embedding(batch_covs_unpacked)
        packed_embedded_covs = nn.utils.rnn.pack_padded_sequence(
            embedded_covs.permute(1, 0, 2), batch.traj_lens,
            enforce_sorted=False
        )
        # eventually may add attention by using the hidden_states/'output' of the GR
        h_0 = self.init_hidden_state.repeat(1, batch_size, 1)
        hidden_states, _ = self.RNN(packed_embedded_covs, h_0)
        unpacked_hidden_states, lengths = self.unpack_and_permute(
            hidden_states, max_len
        )

        static_covs = batch.static_covs
        static_covs[torch.isnan(static_covs)] = -1 
        shaped_static_covs = \
            static_covs.unsqueeze(1).repeat(1, unpacked_hidden_states.shape[1], 1)
        fc_output = self.params_fc_layer(
            torch.cat([unpacked_hidden_states, shaped_static_covs], axis=2)
        )

        #print(fc_output.shape, batch.cov_times.shape)
        if not self.deltas_fixed_to_zero:
            #pred_deltas = torch.exp(-fc_output) - batch.cov_times.unsqueeze(-1)
            pred_deltas = torch.nn.functional.softplus(fc_output, beta=100)
            pred_deltas = pred_deltas - batch.cov_times.unsqueeze(-1)
        else:
            pred_deltas = torch.zeros(all_data.shape[0], 1)
#        print(pred_deltas.shape)

#        next_step_cov_preds = self.make_next_step_cov_preds(
#            unpacked_hidden_states, 
#            batch_covs_unpacked, 
#            lengths
#        ) 

        next_step_cov_preds = torch.zeros(
            unpacked_hidden_states.shape[1] - 1,
            batch_covs_unpacked.shape[0],
            self.params['dynamic_cov_dim'] + 1
        )

        return pred_deltas, unpacked_hidden_states, next_step_cov_preds


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
            self.params['dynamic_cov_dim'] + 1
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

    def unpack_and_permute(self, packed_tensor, max_len):
        unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(
            packed_tensor, total_length=max_len
        )
        unpacked = unpacked.permute(1, 0, 2)
        return unpacked, lens
    
    def set_and_freeze_global_param(self, global_param):
        with torch.no_grad():
            self.global_param_logspace.copy_(
                -torch.log(nn.Parameter(global_param))
            )
        self.freeze_global_param()

    def freeze_global_param(self):
        self.global_param_logspace.requires_grad = False
    def get_global_param(self):
        #if self.distribution_type == 'weibull':
        #    scale = torch.exp(-self.global_param_logspace[0])
        #    # clamp k at two in order to avoid infinite gradients near zero
        #    shape = torch.exp(-self.global_param_logspace[1]) + 2.
        #    return torch.cat([scale.unsqueeze(0), shape.unsqueeze(0)])
        return torch.exp(-self.global_param_logspace)
