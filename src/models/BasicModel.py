import torch
import torch.nn as nn

N_PARAMS_PER_SURVIVAL_DISTRIBUTION = {'GGD': 3}
class BasicModel(nn.module):
    def __init__(self, model_params):
        super(VanillaModel, self).__init__()
        self.params = model_params

        self.RNN = nn.GRU(\
            self.params['covariate_dim'], self.params['hidden_dim'],
            self.params['n_layers_rnn'], dropout=self.params['dropout_prob']
        )

        self.fully_connected_layer = nn.Linear(\
            self.params['hidden_dim'],
            N_PARAMS_PER_SURVIVAL_DISTRIBUTION[self.params['survival_distribution_type']
        )
        


    def forward(self, packed_sequence_batch):
        #TODO: figure out how to make this rnn run the sequences at a batch level, for now
        # it's simpler to do one trajectory at a time
        #predicted_distribution_parameters = torch.zeros([len(covariate_trajectories), N_PARAMS_PER_SURVIVAL_DISTRIBUTION[self.params['survival_distribution_type']]])
        # rnn_input = self.prepare_batches_for_rnn(covariate_trajectory, missing_indicator)
        batch_size = packed_sequence_batch.something
        h_0 = Variable(torch.randn(\
            self.params['n_layers_rnn'], batch_size, 
            self.params['hidden_dim']
        ))
        h, _ = self.RNN(packed_sequence_batch, h_0)
        predicted_distribution_parameters = self.fully_connected_layer(h)
        # THIS GOES IN TRAIN CODE
        #survival_log_loss = self.compute_survival_log_loss()
        #predicted_distribution_parameters[i, :] = predicted_parameters_single_trajectory
        print(predicted_distribution_parameters.shape)
        return predicted_distribution_parameters
      
