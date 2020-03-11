import torch
from torch.autograd import Variable
import torch.nn as nn

SURVIVAL_DISTRIBUTION_CONFIGS = {'ggd': (3, [1, 1, 1]), 'gamma': (2, [1, 1]), 'exp': (1, [1]), 'lnormal':(2, [1, 1])}

N_LAYERS_PER_RNN = 1

class BasicModelOneTheta(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.distribution_type = distribution_type
        # plus one is for the timestamp -> needs to be updated to 2 * covariate_dim + 1 to account for missing indicators
        
        if not self.params['n_layers_rnn'] == None:
            raise ValueError('Basic model assumes a two layer RNN, the first layer represents the encoded history, and the second represents the predicted parameter values. Config for n_layers_rnn must equal None since it cannot change for this model type.')

        # TODO add dropout back in!!
        self.RNN = nn.GRU(\
            self.params['covariate_dim'] + 1, self.params['hidden_dim'],
            #dropout=self.params['dropout']
        )

        self.params_fc_layer = nn.Linear(\
            self.params['hidden_dim'],
            SURVIVAL_DISTRIBUTION_CONFIGS[distribution_type][0]
        ) 

        nn.init.xavier_normal(self.params_fc_layer.weight, gain=.001)
        nn.init.normal_(self.params_fc_layer.bias, std=.00001)

        self.cov_fc_layer1 = nn.Linear(
            self.params['hidden_dim'] + 1,
            self.params['covariate_dim'] * 10 ##TODO fix magic number!!
        )
        self.cov_fc_layer2 = nn.Linear(\
            self.params['covariate_dim'] * 10, #TODO fix magic number!!
            self.params['covariate_dim']
            #self.params['covariate_dim'] + 1 to predict time too
        )
        
    def forward(self, packed_sequence_batch):
        batch_size = packed_sequence_batch.batch_sizes[0]

        h_0_l1 = Variable(torch.randn(\
            N_LAYERS_PER_RNN, batch_size, 
            self.params['hidden_dim']
        ))

        
        h_0_l2 = Variable(torch.randn(\
            N_LAYERS_PER_RNN, batch_size, 
            SURVIVAL_DISTRIBUTION_CONFIGS[self.distribution_type][0]
        ))

        # for LSTM, eventually should make this work in general
        #c_0 = Variable(torch.randn(\
        #    self.params['n_layers_rnn'], batch_size, 
        #    self.params['hidden_dim']
        #))

        # eventually may add attention by using the hidden_states/'output' of the GRU
        output, last_hidden_state = self.RNN(packed_sequence_batch, h_0_l1)

        unpacked_hidden_states, lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        unpacked_hidden_states = unpacked_hidden_states.permute(1, 0, 2)


        #print(unpacked_hidden_states.shape)

        batch_covs_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence_batch)
        batch_covs_unpacked = batch_covs_unpacked.permute(1, 0, 2)

        next_step_cov_preds = self.make_next_step_cov_preds(unpacked_hidden_states, batch_covs_unpacked, lengths) 
        # TODO update the next two lines to predict the params at each step of the hidden output
        # do we have to only feed in the correct hidden state lens here???
        #predicted_distribution_parameters = self.params_fc_layer(last_hidden_state.view(last_hidden_state.shape[1], -1))
        predicted_distribution_parameters =  self.params_fc_layer(last_hidden_state)
        predicted_distribution_parameters = predicted_distribution_parameters.permute(1, 0, 2) 
        ## probably uneccesary
        #predicted_distribution_parameters = self.format_param_preds(predicted_distribution_parameters)

        predicted_distribution_parameters = self.restrict_parameter_ranges(predicted_distribution_parameters)
        #predicted_distribution_parameters.register_hook(self.replace_nans_with_0)
        #self.predicted_distribution_parameters = predicted_distribution_parameters
        #self.batch_covs_time = batch_covs_unpacked[:, :, 0]
        #print(batch_covs_unpacked.shape)
        #print(predicted_distribution_parameters.shape)
        #print(next_step_cov_preds.shape)

        #print(predicted_distribution_parameters)
        #print(unpacked_hidden_states[0:10, 0:3])
        return next_step_cov_preds, predicted_distribution_parameters.squeeze(), lengths

    def format_param_preds(self, predicted_distribution_parameters, lengths):
        pass 
        


    def restrict_parameter_ranges(self, predicted_distribution_parameters):
        are_params_pos_only = SURVIVAL_DISTRIBUTION_CONFIGS[self.distribution_type][1]
        #print(SURVIVAL_DISTRIBUTION_CONFIGS[self.distribution_type])
        restricted_range_params = torch.zeros(predicted_distribution_parameters.shape)
        for i, param_is_pos in enumerate(are_params_pos_only):
            # TODO if/else statements for different distribtuion types
            if param_is_pos:
                restricted_range_params[:, :, i] = torch.exp(-predicted_distribution_parameters[:, :, i])
        return restricted_range_params
                
     
    def replace_nans_with_0(self, grad):
        if not torch.sum(torch.isnan(grad)) == 0:
            print('replaced some gradients that exploded with 0!!')
            #print('nan terms in grad:', torch.sum(torch.isnan(grad)), 'not nan terms in grad', torch.sum(~ torch.isnan(grad)))
            print(grad)
            print(self.predicted_distribution_parameters)
            print(self.batch_covs_time)
            #pass
        grad[torch.isnan(grad)] = torch.tensor([0.])
        grad[torch.isinf(grad)] = torch.tensor([0.])
        if torch.cuda.is_available():
            grad.cuda()
        return torch.autograd.Variable(grad)

    def make_next_step_cov_preds(self, unpacked_hidden_states, batch_covs_unpacked, lengths):
        time_deltas = torch.cat([batch_covs_unpacked[:, 1:, 0], torch.zeros(batch_covs_unpacked.shape[0], 1)], dim=1) - batch_covs_unpacked[:, :, 0]
        #print(time_deltas[0, :], batch_covs_unpacked[0, :, 0])
        unpacked_hidden_states_with_times = torch.cat([unpacked_hidden_states, time_deltas.unsqueeze(2)], dim=2)
        # plus one to try and predict next time step, would also have to update the regularization loss function
        next_step_cov_preds = torch.zeros(unpacked_hidden_states.shape[1] - 1, batch_covs_unpacked.shape[0], self.params['covariate_dim']) 
        
        for batch, (hidden_states_per_step, length) in enumerate(zip(unpacked_hidden_states_with_times, lengths)):
            for i in range(length - 1):
                if self.params['hidden_dim'] == 1:
                    fc_arg = hidden_states_per_step[i].reshape(1, 1)
                
                next_step_cov_preds[i, batch, :] = self.cov_fc_layer2(torch.sigmoid(self.cov_fc_layer1(hidden_states_per_step[i])))
        next_step_cov_preds = next_step_cov_preds.permute(1, 0, 2)

        return next_step_cov_preds

    def freeze_rnn_parameters(self):
        for param in self.RNN.parameters():
            param.requires_grad = False
    
    def freeze_cov_pred_parameters(self):
        for param in self.cov_fc_layer1.parameters():
            param.requires_grad = False
        for param in self.cov_fc_layer2.parameters():
            param.requires_grad = False
