import torch
from  torch.autograd import Variable
import torch.nn as nn
from models.BasicModel import BasicModel
N_PARAMS_PER_SURVIVAL_DISTRIBUTION = {'ggd': 3, 'gamma': 2, 'exp': 2, 'lnormal':2}
class GlobalPlusEpsModel(nn.Module):
    def __init__(self, model_params, distribution_type):
        super().__init__()
        self.params = model_params
        self.basic_model = BasicModel(model_params, distribution_type)
        # plus one is for the timestamp -> needs to be updated to 2 * covariate_dim + 1 to account for missing indicators
        self.RNN = nn.GRU(\
            self.params['covariate_dim'] + 1, self.params['hidden_dim'],
            self.params['n_layers_rnn'] #dropout=self.params['dropout']
        )

        self.params_fc_layer = nn.Linear(\
            self.params['hidden_dim'] * self.params['n_layers_rnn'],
            N_PARAMS_PER_SURVIVAL_DISTRIBUTION[distribution_type]
        )

        self.cov_fc_layer = nn.Linear(\
            self.params['hidden_dim'],
            self.params['covariate_dim'] + 1
        )
        #for parameter in self.parameters():
        #    parameter = (parameter < 0)  * 0. + (parameter > 0) * parameter
        #    print(parameter)
        


    def forward(self, packed_sequence_batch):
        #TODO: figure out how to make this rnn run the sequences at a batch level, for now
        # it's simpler to do one trajectory at a time
        batch_size = packed_sequence_batch.batch_sizes[0]
        h_0 = Variable(torch.randn(\
            self.params['n_layers_rnn'], batch_size, 
            self.params['hidden_dim']
        ))
        c_0 = Variable(torch.randn(\
            self.params['n_layers_rnn'], batch_size, 
            self.params['hidden_dim']
        ))
        # eventually may add attention by using the hidden_states/'output' of the GRU
        output, last_hidden_state = self.RNN(packed_sequence_batch, h_0)
        #             last_hidden_state 
        unpacked_hidden_states, lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        
        #next_step_cov_preds = torch.zeros(unpacked_hidden_states.shape[0] - 1, batch_size, self.params['covariate_dim'] + 1)
        #for batch, (hidden_states_per_step, length) in enumerate(zip(unpacked_hidden_states.reshape(unpacked_hidden_states.shape[1], unpacked_hidden_states.shape[0], self.params['hidden_dim']), lengths)):
        #    for i in range(length - 1):
        #        if self.params['hidden_dim'] == 1:
        #            fc_arg = hidden_states_per_step[i].reshape(1, 1)
        #        next_step_cov_preds[i, batch, :] = self.cov_fc_layer(hidden_states_per_step[i])
            
        
        predicted_distribution_parameters = self.params_fc_layer(last_hidden_state.view(last_hidden_state.shape[1], -1))
        predicted_distribution_parameters_with_sigma_positive = torch.zeros(predicted_distribution_parameters.shape[0], predicted_distribution_parameters.shape[1])
        predicted_distribution_parameters_with_sigma_positive[:, 1] = torch.abs(predicted_distribution_parameters[:, 1])
        if self.params['survival_distribution_type'] == 'GGD':
            predicted_distribution_parameters_with_sigma_positive[:, 0] = predicted_distribution_parameters[:, 0]
            predicted_distribution_parameters_with_sigma_positive[:, 2] = torch.abs(predicted_distribution_parameters[:, 2])
        else:
            predicted_distribution_parameters_with_sigma_positive[:, 0] = torch.abs(predicted_distribution_parameters[:, 0])
        #print(predicted_distribution_parameters_with_sigma_positive[0:30, 0])
        predicted_distribution_parameters_with_sigma_positive.register_hook(self.replace_nans_with_0)   
        global_params = torch.mean(predicted_distribution_parameters_with_sigma_positive)
        next_step_cov_preds, local_params = self.basic_model(packed_sequence_batch)
        return next_step_cov_preds, global_params + local_params
     
    def replace_nans_with_0(self, grad):
        if not torch.sum(torch.isnan(grad)) == 0:
            #print('replaced some gradients that exploded with 0!!')
            #print('nan terms in grad:', torch.sum(torch.isnan(grad)), 'not nan terms in grad', torch.sum(~ torch.isnan(grad)))
            pass
        grad[torch.isnan(grad)] = torch.tensor([0.])
        grad[torch.isinf(grad)] = torch.tensor([0.])
        if torch.cuda.is_available():
            grad.cuda()
        return torch.autograd.Variable(grad)
 
