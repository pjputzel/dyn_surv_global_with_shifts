import torch
import torch.nn as nn

N_PARAMS_PER_SURVIVAL_DISTRIBUTION = {'GGD': 3}
class VanillaModel(nn.module):
    def __init__(self, model_params):
        super(VanillaModel, self).__init__()
        self.params = model_params

        self.RNN = nn.GRU(\
            self.params['covariate_dim'], self.params['hidden_dim'],
            self.params['n_layers'], dropout=self.params['dropout_prob']
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
        h_0 = Variable(torch.randn(\
            self.params['n_layers'], self.params['batch_size'], 
            self.params['hidden_size']
        ))
        h, _ = self.RNN(packed_sequence_batch, h_0)
        predicted_distribution_parameters = self.fully_connected_layer(h)
        # THIS GOES IN TRAIN CODE
        #survival_log_loss = self.compute_survival_log_loss()
        #predicted_distribution_parameters[i, :] = predicted_parameters_single_trajectory
        return predicted_distribution_parameters
      
    # TODO move this function to a trainer, not inside the model!!! Model
    # should only handle a single packed sequence at a time!
    def prepare_sequences_for_rnn(self, covariate_trajectories, missing_indicators):
        # first we need to pad the sequences to the max trajectory length
        padded_trajectories, trajectory_lengths = self.pad_data_with_zeros(covariate_trajectories)

        padded_trajectories = torch.tensor(padded_trajectories)
        trajectory_lengths = torch.tensor(trajectory_lengths)
        batches_of_padded_sequences = []
        for batch in range((padded_trajectories.shape[0]//self.params['batch_size']) + 1):
            batch_input = padded_trajectories[batch * self.params['batch_size']: (batch + 1) * self.params['batch_size']]
            batch_trajectory_lengths = trajectory_lengths[batch * self.params['batch_size']: (batch + 1) * self.params['batch_size']]
            batch_input = batch_input.view(padded_trajectories.shape[1], self.params['batch_size'], 1 if len(padded_trajectories.shape) == 2 else padded_trajectories.shape[2])
            rnn_input = torch.nn.pack_padded_sequence(batch_input, batch_trajectory_lengths)
            batches_of_padded_sequences.append(batch)
        return batches_of_padded_sequences
        
    
    def pad_data_with_zeros(self, covariate_trajectories): 
        max_len_trajectory = np.max([len(traj) for traj in covariate_trajectories])
        padded_trajectories = []
        trajectory_lengths = []
        for traj in covariate_trajectories:
            if len(traj) < max_len_trajectory:
                padded_trajectory = traj + [0 for i in range(max_len_trajectory) - len(traj)]
            padded_trajectories.append(padded_trajectory)
            trajectory_lengths.append(len(traj))
        return padded_trajectories, trajectory_lengths
