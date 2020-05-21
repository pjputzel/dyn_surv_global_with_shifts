from data_loading.SyntheticDataLoader import SyntheticDataLoader
import numpy as np
import torch

# there should only be one data-input (don't subclass) but one dataloader per new dataset

### DataInput loads the data, and prepares the data for input into different parts of the pipeline

class DataInput:

    def __init__(self, data_input_params):
        self.params = data_input_params
    
    def load_data(self):
        if self.params['dataset_name'] == 'synth':
            dataloader = SyntheticDataLoader(self.params['data_loading_params'])
            # expected format for covariate trajectories: list of [ [timestamp, [covariate_dim1, covariate_dim2, .... ]] ]
            self.event_times, self.censoring_indicators, self.missing_indicators, self.covariate_trajectories = dataloader.load_data()
            self.unshuffled_idxs = torch.arange(len(self.event_times))
        else:
            raise ValueError('Dataset name %s not recognized' %self.params['dataset_name'])

        self.format_data() 

    def format_data(self):
        self.format_cov_trajs_and_event_times_time_rep()
        # TODO: format the missing indicators and 
        # any interpolation here
        # self.format_cov_trajs_missingness()

        self.pad_cov_trajs_with_zeros()
        self.convert_to_tensors()

    def format_cov_trajs_and_event_times_time_rep(self):
        # real data doesn't include true times, only the differences are
        # accurate to help de-identify patients
        # so shift everything to be relative to first covariate measurement time
        cov_times = [
            [cov_event[0] for cov_event in traj]
            for traj in self.covariate_trajectories
        ]
        self.covariate_trajectories = [ 
            [ [event[0] - cov_times[i][0]] + event[1] for event in traj] 
            for i, traj in enumerate(self.covariate_trajectories)
        ]
        
        self.event_times = [
            event_time - cov_times[i][0]
            for i, event_time in enumerate(self.event_times)
        ]

        cov_time_rep = self.params['cov_time_representation']
        if cov_time_rep == 'delta':
            self.covariate_trajectories = [ 
                [ 
                    [event[0] - cov_times[i][j - 1]] + event[1] if j > 0 else traj
                    for j, event in enumerate(traj)
                ] 
                for i, traj in enumerate(self.covariate_trajectories)
            ]


        elif cov_time_rep == 'absolute':
            # don't need to do anything in this case
            pass
        else:
            message = 'Time representation %s not defined' %cov_time_rep
            raise ValueError(message)


    def pad_cov_trajs_with_zeros(self): 
        max_len_trajectory = np.max([len(traj) for traj in self.covariate_trajectories])
        padded_trajectories = []
        trajectory_lengths = []
        for traj in self.covariate_trajectories:
            if len(traj) < max_len_trajectory:
                padded_trajectory = traj + 
                    [
                        [0 for i in range(len(traj[0]))] 
                        for i in range(max_len_trajectory - len(traj))
                    ]
            else:
                padded_trajectory = traj
            padded_trajectories.append(padded_trajectory)
            trajectory_lengths.append(len(traj))
        self.covariate_trajectories = padded_trajectories
        self.trajectory_lengths = trajectory_lengths
    
    def convert_data_to_tensors(self):
        # could make float64/32 an option in params
        self.covariate_trajectories = torch.tensor(self.covariate_trajectories, dtype=torch.float64)
        self.trajectory_lengths = torch.tensor(self.trajectory_lengths, dtype=torch.float64)
        self.event_times = torch.tensor(self.event_times, dtype=torch.float64)
        self.censoring_indicators = torch.tensor(self.censoring_indicators)
        self.missing_indicators = torch.tensor(self.missing_indicators)

    def make_randomized_batches(self, batch_size):
        self.shuffle_all_data()

        batches = []
        if self.covariate_trajectories.shape[0] % batch_size == 0:
            num_batches = self.covariate_trajectories.shape[0]//batch_size
        else:
            num_batches = self.covariate_trajectories.shape[0]//batch_size + 1

        for batch_idx in range(num_batches):
            batch = Batch(*self.get_batch_data(batch_idx))
            batches.append(batch) 
        self.batches = batches
    

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
        self.trajectory_lengths = [self.trajectory_lengths[idx] for idx in idxs]

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
        self.trajectory_lengths = self.trajectory_lengths[idx for idx in idxs]
    
    def get_batch_data(self, batch_idx):
        batch_indices = batch_idx * batch_size: (batch_idx + 1) * batch_size

        batch_cov_trajs = self.covariate_trajectories[batch_indices]
        batch_traj_lengths = self.trajectory_lengths[batch_indices]
        batch_cov_trajs = batch_cov_trajs.permute(1, 0, 2)
        batch_packed_cov_trajs = torch.nn.utils.rnn.pack_padded_sequence(
            batch_cov_trajs, batch_traj_lengths, enforce_sorted=False
        )
        batch_event_times = self.batch_event_times[batch_indices]
        batch_censoring_indicators = self.censoring_indicators[batch_indices]
        batch_unshuffle_idxs = self.unshuffled_idxs[batch_indices]
        return 
            batch_packed_cov_trajs, batch_event_times, batch_censoring_indicators, 
            batch_traj_lengths, batch_unshuffle_idxs
            

# Simple helper class to make passing around batches of the data easier
# also handles unpacking cov trajs
class Batch:

    def __init__(self,
        batch_packed_cov_trajs, batch_event_times, batch_censoring_indicators,
        batch_traj_lengths, batch_unshuffle_idxs
    ):

        self.packed_cov_trajs = batch_cov_trajs
        self.event_times = batch_event_times
        self.censoring_indicators = batch_censoring_indicators
        self.trajectory_lengths = batch_traj_lengths
        self.unshuffled_idxs = batch_unshuffle_idxs

    def get_unpacked_padded_cov_trajs(self):
        #TODO: implement this using rnn utils from pytorch
        pass

    
