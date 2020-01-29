# there should only be one data-input (don't subclass) but one dataloader per new dataset

### DataInput loads the data, and prepares the data for input into different parts of the pipeline

class DataInput:

    def __init__(self, data_input_params):
        self.data_input_params = data_input_params
    
    def load_data(self):
        if self.data_input_params['dataset_name'] == 'synthetic':
            dataloader = SyntheticDataLoader(self.data_input_params['data_loading_params'])
            self.event_times, self.censoring_indicators, self.missing_indicators, self.covariate_trajectories = dataloader.load_data()
        else:
            raise ValueError('Dataset name %s not recognized' %self.data_input_params['dataset_name'])

    def prepare_sequences_for_rnn(self, batch_size):
        # first we need to pad the sequences to the max trajectory length
        padded_trajectories, trajectory_lengths = self.pad_data_with_zeros()

        padded_trajectories = torch.tensor(padded_trajectories)
        trajectory_lengths = torch.tensor(trajectory_lengths)
        batches_of_padded_sequences = []
        for batch in range((padded_trajectories.shape[0]//batch_size) + 1):
            batch_input = padded_trajectories[batch * batch_size: (batch + 1) * batch_size]
            batch_trajectory_lengths = trajectory_lengths[batch * batch_size: (batch + 1) * batch_size]]
            batch_input = batch_input.view(padded_trajectories.shape[1], batch_size, 1 if len(padded_trajectories.shape) == 2 else padded_trajectories.shape[2])
            rnn_input = torch.nn.pack_padded_sequence(batch_input, batch_trajectory_lengths)
            batches_of_padded_sequences.append(batch)
        self.batches_of_padded_sequences = batches_of_padded_sequences
        return batches_of_padded_sequences
        
    
    def pad_data_with_zeros(self): 
        max_len_trajectory = np.max([len(traj) for traj in self.covariate_trajectories])
        padded_trajectories = []
        trajectory_lengths = []
        for traj in self.covariate_trajectories:
            if len(traj) < max_len_trajectory:
                padded_trajectory = traj + [0 for i in range(max_len_trajectory) - len(traj)]
            padded_trajectories.append(padded_trajectory)
            trajectory_lengths.append(len(traj))
        return padded_trajectories, trajectory_lengths

