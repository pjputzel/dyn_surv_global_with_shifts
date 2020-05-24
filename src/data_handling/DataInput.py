from data_handling.SyntheticDataLoader import SyntheticDataLoader
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
        self.convert_data_to_tensors()

    def format_cov_trajs_and_event_times_time_rep(self):
        # real data doesn't include true times, only the differences are
        # accurate to help de-identify patients
        # so shift everything to be relative to first covariate measurement time
        cov_times_abs = [
            [cov_event[0] for cov_event in traj]
            for traj in self.covariate_trajectories
        ]
        self.covariate_trajectories = [ 
            [[cov_event[0] - cov_times_abs[i][0]] + cov_event[1] for cov_event in traj]
            for i, traj in enumerate(self.covariate_trajectories)
        ]
        
        
        self.event_times = [
            event_time - cov_times_abs[i][0]
            for i, event_time in enumerate(self.event_times)
        ]
        
        self.cov_times = [
            [time - traj_cov_times[0] for time in traj_cov_times] 
            for traj_cov_times in cov_times_abs
        ]
        

        cov_time_rep = self.params['cov_time_representation']
        if cov_time_rep == 'delta':
            # event[0] is still the time of the covariate measurement
            self.covariate_trajectories = [ 
                [ 
                    [event[0] - self.cov_times[i][j - 1]] + event[1:]  if j > 0 else event
                    for j, event in enumerate(traj)
                ] 
                for i, traj in enumerate(self.covariate_trajectories)
            ]
            for i, traj in enumerate(self.covariate_trajectories):
                for j, cov_event in enumerate(traj):
                    if not j ==0:
                        self.covariate_trajectories[i][j][0] = self.cov_times[i][j] - self.cov_times[i][j - 1]

                        

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
        padded_cov_times = []
        padding_indicators = []
        for traj in self.covariate_trajectories:
            padding_indicators_traj = [0 for i in range(len(traj))]
            if len(traj) < max_len_trajectory:
                padded_trajectory = traj + \
                    [
                        [0 for i in range(len(traj[0]))] 
                        for i in range(max_len_trajectory - len(traj))
                    ]
                padding_indicators_traj.extend(
                    [1 for i in range(max_len_trajectory - len(traj))]
                )
            else:
                padded_trajectory = traj
            
            # note padding is zero here as well
            # so when checking for padding make sure
            # to consider that the first entry is zero as well
            padded_cov_times.append(
                [
                    cov_event[0] for cov_event in padded_trajectory
                ]
            )
            padding_indicators.append(padding_indicators_traj)

            
            padded_trajectories.append(padded_trajectory)
            trajectory_lengths.append(len(traj))
        self.covariate_trajectories = padded_trajectories
        self.trajectory_lengths = trajectory_lengths
        self.cov_times = padded_cov_times
        self.max_len_trajectory = max_len_trajectory
        self.padding_indicators = padding_indicators
    
    def convert_data_to_tensors(self):
        # could make float64/32 an option in params
        self.covariate_trajectories = torch.tensor(self.covariate_trajectories, dtype=torch.float64)
        self.trajectory_lengths = torch.tensor(self.trajectory_lengths, dtype=torch.float64)
        self.event_times = torch.tensor(self.event_times, dtype=torch.float64)
        self.censoring_indicators = torch.tensor(self.censoring_indicators)
        self.missing_indicators = torch.tensor(self.missing_indicators)
        self.cov_times = torch.tensor(self.cov_times)
        self.padding_indicators = torch.tensor(self.padding_indicators)

    def make_randomized_batches(self, batch_size):
        self.shuffle_all_data()

        batches = []
        num_individuals = self.covariate_trajectories.shape[0]
        if num_individuals % batch_size == 0:
            num_batches = num_individuals//batch_size
        else:
            num_batches = num_individuals//batch_size + 1
        
        for batch_idx in range(num_batches):
            batch = Batch(*self.get_batch_data(batch_idx, batch_size), int(self.max_len_trajectory))
            batches.append(batch) 
        self.batches = batches
    

    def shuffle_all_data(self):
        # to avoid being unable to unshuffle-> makes
        # sure they aren't two shuffles in a row
        # we want to be able to see original order for analysis
        self.unshuffle_all_data()
        idxs = torch.randperm(len(self.covariate_trajectories))
        self.covariate_trajectories = self.covariate_trajectories[idxs]
        self.missing_indicators = self.missing_indicators[idxs]
        self.censoring_indicators = self.censoring_indicators[idxs]
        self.event_times = self.event_times[idxs]
        self.trajectory_lengths = self.trajectory_lengths[idxs]
        self.cov_times = self.cov_times[idxs]
        self.padding_indicators = self.padding_indicators[idxs]
#        self.covariate_trajectories = [self.covariate_trajectories[idx] for idx in idxs]
#        self.missing_indicators = [self.missing_indicators[idx] for idx in idxs]
#        self.censoring_indicators = [self.censoring_indicators[idx] for idx in idxs]
#        self.event_times = [self.event_times[idx] for idx in idxs]
#        self.trajectory_lengths = [self.trajectory_lengths[idx] for idx in idxs]

        self.shuffled_idxs = idxs
        for i, idx in enumerate(idxs):
            self.unshuffled_idxs[idx] = i
        #print(self.shuffled_idxs, self.unshuffled_idxs)

    def unshuffle_all_data(self):
        idxs = self.unshuffled_idxs
        self.covariate_trajectories = self.covariate_trajectories[idxs]
        self.missing_indicators = self.missing_indicators[idxs]
        self.censoring_indicators = self.censoring_indicators[idxs]
        self.event_times = self.event_times[idxs]
        self.trajectory_lengths = self.trajectory_lengths[idxs]
        self.cov_times = self.cov_times[idxs]
        self.padding_indicators = self.padding_indicators[idxs]
#        self.covariate_trajectories = [self.covariate_trajectories[idx] for idx in idxs]
#        self.missing_indicators = [self.missing_indicators[idx] for idx in idxs]
#        self.censoring_indicators = [self.censoring_indicators[idx] for idx in idxs]
#        self.event_times = [self.event_times[idx] for idx in idxs]
#        self.trajectory_lengths = [self.trajectory_lengths[idx] for idx in idxs]
    
    def get_batch_data(self, batch_idx, batch_size):
        batch_indices = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)

        batch_cov_trajs = self.covariate_trajectories[batch_indices]
        batch_traj_lengths = self.trajectory_lengths[batch_indices]
        batch_cov_trajs = batch_cov_trajs.permute(1, 0, 2)
        batch_packed_cov_trajs = torch.nn.utils.rnn.pack_padded_sequence(
            batch_cov_trajs, batch_traj_lengths, enforce_sorted=False
        )
        batch_event_times = self.event_times[batch_indices]
        batch_censoring_indicators = self.censoring_indicators[batch_indices]
        batch_unshuffle_idxs = self.unshuffled_idxs[batch_indices]
        batch_cov_times = self.cov_times[batch_indices]
        return \
            batch_packed_cov_trajs, batch_cov_times, batch_event_times,\
            batch_censoring_indicators, batch_traj_lengths, batch_unshuffle_idxs
            

# Simple helper class to make passing around batches of the data easier
# also handles unpacking cov trajs
# note: packed_cov_trajs uses whatever time rep is selected for cov_times
# while cov_times is the times relative to first covariate event
# for example: packed cov trajs may have time deltas between cov measurements,
# while cov times will be different and just have times relative to first
# event
class Batch:

    def __init__(self,
        batch_packed_cov_trajs, batch_cov_times, batch_event_times, 
        batch_censoring_indicators, batch_traj_lengths, batch_unshuffle_idxs,
        max_seq_len_all_batches
    ):

        self.packed_cov_trajs = batch_packed_cov_trajs
        self.cov_times = batch_cov_times
        self.event_times = batch_event_times
        self.censoring_indicators = batch_censoring_indicators
        self.trajectory_lengths = torch.tensor(batch_traj_lengths, dtype=torch.float64)
        self.unshuffled_idxs = batch_unshuffle_idxs
        self.max_seq_len_all_batches = max_seq_len_all_batches

    def get_unpacked_padded_cov_trajs(self):
        batch_covs, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            self.packed_cov_trajs, total_length=self.max_seq_len_all_batches
        )

        batch_covs = batch_covs.transpose(0,1)
        return batch_covs
