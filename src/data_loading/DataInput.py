from data_loading.SyntheticDataLoader import SyntheticDataLoader
import numpy as np
import torch

# there should only be one data-input (don't subclass) but one dataloader per new dataset

### DataInput loads the data, and prepares the data for input into different parts of the pipeline

class DataInput:

    def __init__(self, data_input_params):
        self.data_input_params = data_input_params
    
    def load_data(self):
        if self.data_input_params['dataset_name'] == 'synth':
            dataloader = SyntheticDataLoader(self.data_input_params['data_loading_params'])
            # expected format for covariate trajectories: list of [ [timestamp, [covariate_dim1, covariate_dim2, .... ]] ]
            self.event_times, self.censoring_indicators, self.missing_indicators, self.covariate_trajectories = dataloader.load_data()
            self.unshuffled_idxs = torch.arange(len(self.event_times))
        else:
            raise ValueError('Dataset name %s not recognized' %self.data_input_params['dataset_name'])

    def prepare_sequences_for_rnn(self, batch_size):
        # transform trajectories to have time concatenated into the first spot of each observation
        # eventually will also need to concatenate missing indicators per covariate dim as well
        self.shuffle_all_data()
        self.covariate_trajectories_with_time = [ [ [event[0]] + event[1] for event in traj] for traj in self.covariate_trajectories]


        # first we need to pad the sequences to the max trajectory length
        padded_trajectories, trajectory_lengths = self.pad_data_with_zeros()
        padded_trajectories = torch.tensor(padded_trajectories, dtype=torch.float64)
        trajectory_lengths = torch.tensor(trajectory_lengths, dtype=torch.float64)



        batches_of_padded_sequences = []
        # note batch idxs is into the shuffled array
        #batch_idxs = []
        if padded_trajectories.shape[0] % batch_size == 0:
            num_batches = padded_trajectories.shape[0]//batch_size
        else:
            num_batches = padded_trajectories.shape[0]//batch_size + 1

        for batch in range(num_batches):
            batch_input = padded_trajectories[batch * batch_size: (batch + 1) * batch_size]
            batch_trajectory_lengths = trajectory_lengths[batch * batch_size: (batch + 1) * batch_size]
            # instead of -1 I had batch size before- TODO: double check this!!
            #print(batch_input.shape)
            #print(batch_input[0])
            batch_input = batch_input.permute(1, 0, 2)
            #batch_input = batch_input.view(batch_input.shape[1], batch_input.shape[0], batch_input.shape[2])
            #print(batch_input.shape)
            #print(batch_trajectory_lengths)
            #print(batch_input[:, 0, :])
            rnn_input = torch.nn.utils.rnn.pack_padded_sequence(batch_input, batch_trajectory_lengths, enforce_sorted=False)
            #print(rnn_input)
            batches_of_padded_sequences.append(rnn_input)
        self.batches_of_padded_sequences = batches_of_padded_sequences
        return batches_of_padded_sequences
    
    def shuffle_all_data(self):
        # to avoid being unable to unshuffle-> makes
        # sure they aren't two shuffles in a row
        # we want to be able to see original order for analysis
        self.unshuffle_all_data()
        idxs = torch.randperm(len(self.covariate_trajectories))
        self.covariate_trajectories = [self.covariate_trajectories[idx] for idx in idxs]
        self.missing_indicators = [self.missing_indicators[idx] for idx in idxs]
        self.censoring_indicators = [self.censoring_indicators[idx] for idx in idxs]
        self.event_times = [self.event_times[idx] for idx in idxs]

        self.shuffled_idxs = idxs
        for i, idx in enumerate(idxs):
            self.unshuffled_idxs[idx] = i
        #print(self.shuffled_idxs, self.unshuffled_idxs)

    def unshuffle_all_data(self):
        idxs = self.unshuffled_idxs
        self.covariate_trajectories = [self.covariate_trajectories[idx] for idx in idxs]
        self.missing_indicators = [self.missing_indicators[idx] for idx in idxs]
        self.censoring_indicators = [self.censoring_indicators[idx] for idx in idxs]
        self.event_times = [self.event_times[idx] for idx in idxs]
        
    
    def pad_data_with_zeros(self): 
        max_len_trajectory = np.max([len(traj) for traj in self.covariate_trajectories])
        padded_trajectories = []
        trajectory_lengths = []
        for traj in self.covariate_trajectories_with_time:
            if len(traj) < max_len_trajectory:
                padded_trajectory = traj + [[0 for i in range(len(traj[0]))] for i in range(max_len_trajectory - len(traj))]
            else:
                padded_trajectory = traj
            padded_trajectories.append(padded_trajectory)
            trajectory_lengths.append(len(traj))
        return padded_trajectories, trajectory_lengths
    
