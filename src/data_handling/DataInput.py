from data_handling.SyntheticDataLoader import SyntheticDataLoader
from pandas import qcut
from sklearn.model_selection import train_test_split
from data_handling.DmCvdDataLoader import DmCvdDataLoader
import numpy as np
import torch

# there should only be one data-input (don't subclass) but one dataloader per new dataset
# DataInput is agnostic to tr/te split, but the made batches which are used by the rest of the model will use the corresponding training/testing idxs
### DataInput loads the data, and prepares the data for input into different parts of the pipeline

class DataInput:

    def __init__(self, data_input_params):
        self.params = data_input_params
    
    def load_data(self):
        if self.params['dataset_name'] == 'synth':
            dataloader = SyntheticDataLoader(self.params['data_loading_params'])
            # expected format for covariate trajectories: list of [ [timestamp, [covariate_dim1, covariate_dim2, .... ]] ]
        elif self.params['dataset_name'] == 'dm_cvd':
            dataloader = DmCvdDataLoader(self.params['data_loading_params'])
        else:
            raise ValueError('Dataset name %s not recognized' %self.params['dataset_name'])

        self.event_times, self.censoring_indicators, self.missing_indicators, self.covariate_trajectories, self.static_covs = dataloader.load_data()
        self.format_data()
        self.normalize_data()
        self.split_data()
        self.unshuffled_tr_idxs = torch.arange(len(self.event_times_tr))
#        self.unshuffled_tr_idxs = torch.arange(len(self.event_times_tr))
        #print(self.unshuffled_tr_idxs)

    def normalize_data(self):
        # normalization is needed for labtests in cov trajectories only
        LABTEST_CUTOFF = 120  # plus one from notebook since includes timedeltas
        cov_trajs = self.covariate_trajectories
        print(cov_trajs.shape)
        #cov_trajs[torch.isnan(cov_trajs)] = 0
        min_covs = torch.min(torch.min(cov_trajs[:, :, 1:120], axis=0)[0], axis=0)[0]
        max_covs = torch.max(torch.max(cov_trajs[:, :, 1:120], axis=0)[0], axis=0)[0]
#        print('min covs:' , min_covs)
#        print('max covs:', max_covs)
        numerator = cov_trajs[:, :, 1:120] - min_covs
        denominator =  max_covs - min_covs 
        #print(numerator/denominator)
        normalized_labtests = numerator/denominator
        normalized_labtests[torch.isnan(normalized_labtests)] = 0
#        print('Normalized labtests:', torch.sum(normalized_labtests, axis=1)[0])
        self.covariate_trajectories[:, :, 1:120] = normalized_labtests

    def format_data(self):
        self.format_cov_trajs_and_event_times_time_rep()
        # TODO: format the missing indicators and 
        # any interpolation here
        # self.format_cov_trajs_missingness()
        # ended up doing this in preprocessing

        self.pad_cov_trajs_with_zeros()
        self.convert_data_to_tensors()
        self.concatenate_covs_with_missingness()

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
            #for i, traj in enumerate(self.covariate_trajectories):
            #    for j, cov_event in enumerate(traj):
            #        if not j ==0:
            #            self.covariate_trajectories[i][j][0] = self.cov_times[i][j] - self.cov_times[i][j - 1]

                        

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
        padded_missing_indicators = []
        for i, traj in enumerate(self.covariate_trajectories):
            padding_indicators_traj = [0 for i in range(len(traj))]
            if len(traj) < max_len_trajectory:
                padded_trajectory = traj + \
                    [
                        [0 for i in range(len(traj[0]))] 
                        for i in range(max_len_trajectory - len(traj))
                    ]
                padded_missing_indicator = self.missing_indicators[i] +\
                    [
                        [0 for i in range(len(traj[0]) - 1)] 
                        for i in range(max_len_trajectory - len(traj))
                    ]
                padding_indicators_traj.extend(
                    [1 for i in range(max_len_trajectory - len(traj))]
                )
            else:
                padded_trajectory = traj
                padded_missing_indicator = self.missing_indicators[i] 
            # note padding is zero here as well
            # so when checking for padding make sure
            # to consider that the first entry is zero as well
            padded_cov_times.append(
                [
                    cov_event[0] for cov_event in padded_trajectory
                ]
            )
            padding_indicators.append(padding_indicators_traj)

            #print(len(padded_trajectory[0]), len(padded_missing_indicator[0]))
            padded_trajectories.append(padded_trajectory)
            padded_missing_indicators.append(padded_missing_indicator)
            trajectory_lengths.append(len(traj))
        self.covariate_trajectories = padded_trajectories
        self.trajectory_lengths = trajectory_lengths
        self.cov_times = padded_cov_times
        self.max_len_trajectory = max_len_trajectory
        self.padding_indicators = padding_indicators
        self.missing_indicators = padded_missing_indicators
   
 
    def convert_data_to_tensors(self):
        # could make float64/32 an option in params
        self.covariate_trajectories = torch.tensor(self.covariate_trajectories, dtype=torch.float64)
        self.trajectory_lengths = torch.tensor(self.trajectory_lengths, dtype=torch.float64)
        self.event_times = torch.tensor(self.event_times, dtype=torch.float64)
        self.censoring_indicators = torch.tensor(self.censoring_indicators)
        self.missing_indicators = torch.tensor(self.missing_indicators)
        self.cov_times = torch.tensor(self.cov_times)
        self.padding_indicators = torch.tensor(self.padding_indicators)
        self.static_covs = torch.tensor(self.static_covs)

    def concatenate_covs_with_missingness(self):
        self.covariate_trajectories = torch.cat([self.covariate_trajectories, self.missing_indicators], axis=-1)

    def split_data(self):
        # just make *_tr and *_te based on a split
        # if you want to do CV with disjoint sets each step then I'd make a separate function
        # which returns an iterator over the CV splits return tr/te batches for each split
        # accordingly, also this will of course be its own main
        te_percent = self.params['te_percent']
        self.tr_idxs, self.te_idxs = train_test_split(
            torch.arange(self.event_times.shape[0]), test_size=te_percent
        )

        self.covariate_trajectories_tr = self.covariate_trajectories[self.tr_idxs]
        self.trajectory_lengths_tr = self.trajectory_lengths[self.tr_idxs]
        self.event_times_tr = self.event_times[self.tr_idxs]
        self.censoring_indicators_tr = self.censoring_indicators[self.tr_idxs]
        self.missing_indicators_tr = self.missing_indicators[self.tr_idxs]
        self.cov_times_tr = self.cov_times[self.tr_idxs]
        self.padding_indicators_tr = self.padding_indicators[self.tr_idxs]
        self.static_covs_tr = self.static_covs[self.tr_idxs]
        
        self.covariate_trajectories_te = self.covariate_trajectories[self.te_idxs]
        self.trajectory_lengths_te = self.trajectory_lengths[self.te_idxs]
        self.event_times_te = self.event_times[self.te_idxs]
        self.censoring_indicators_te = self.censoring_indicators[self.te_idxs]
        self.missing_indicators_te = self.missing_indicators[self.te_idxs]
        self.cov_times_te = self.cov_times[self.te_idxs]
        self.padding_indicators_te = self.padding_indicators[self.te_idxs]
        self.static_covs_te = self.static_covs[self.te_idxs]

    def make_randomized_tr_batches(self, batch_size):
        self.shuffle_tr_data()
        
        
        batches = []
#        num_individuals = self.covariate_trajectories.shape[0]
        num_individuals_tr = len(self.tr_idxs)
        if num_individuals_tr % batch_size == 0:
            num_batches = num_individuals_tr//batch_size
        else:
            num_batches = num_individuals_tr//batch_size + 1
        
        for batch_idx in range(num_batches):
            batch = Batch(*self.get_tr_batch_data(batch_idx, batch_size), int(self.max_len_trajectory))
            batches.append(batch) 
        self.tr_batches = batches
    

    def shuffle_tr_data(self):
        # to avoid being unable to unshuffle-> makes
        # sure they aren't two shuffles in a row
        # we want to be able to see original order for analysis
        self.unshuffle_tr_data()
        idxs = torch.randperm(len(self.covariate_trajectories_tr))
        self.covariate_trajectories_tr = self.covariate_trajectories_tr[idxs]
        self.missing_indicators_tr = self.missing_indicators_tr[idxs]
        self.censoring_indicators_tr = self.censoring_indicators_tr[idxs]
        self.event_times_tr = self.event_times_tr[idxs]
        self.trajectory_lengths_tr = self.trajectory_lengths_tr[idxs]
        self.cov_times_tr = self.cov_times_tr[idxs]
        self.padding_indicators_tr = self.padding_indicators_tr[idxs]
        self.static_covs_tr = self.static_covs_tr[idxs]
#        self.covariate_trajectories = [self.covariate_trajectories[idx] for idx in idxs]
#        self.missing_indicators = [self.missing_indicators[idx] for idx in idxs]
#        self.censoring_indicators = [self.censoring_indicators[idx] for idx in idxs]
#        self.event_times = [self.event_times[idx] for idx in idxs]
#        self.trajectory_lengths = [self.trajectory_lengths[idx] for idx in idxs]

        self.shuffled_tr_idxs = idxs
        for i, idx in enumerate(idxs):
            self.unshuffled_tr_idxs[idx] = i
        #print(self.shuffled_idxs, self.unshuffled_idxs)

    def unshuffle_tr_data(self):
        idxs = self.unshuffled_tr_idxs
        self.covariate_trajectories_tr = self.covariate_trajectories_tr[idxs]
        self.missing_indicators_tr = self.missing_indicators_tr[idxs]
        self.censoring_indicators_tr = self.censoring_indicators_tr[idxs]
        self.event_times_tr = self.event_times_tr[idxs]
        self.trajectory_lengths_tr = self.trajectory_lengths_tr[idxs]
        self.cov_times_tr = self.cov_times_tr[idxs]
        self.padding_indicators_tr = self.padding_indicators_tr[idxs]
        self.static_covs_tr = self.static_covs_tr[idxs]
#        self.covariate_trajectories = [self.covariate_trajectories[idx] for idx in idxs]
#        self.missing_indicators = [self.missing_indicators[idx] for idx in idxs]
#        self.censoring_indicators = [self.censoring_indicators[idx] for idx in idxs]
#        self.event_times = [self.event_times[idx] for idx in idxs]
#        self.trajectory_lengths = [self.trajectory_lengths[idx] for idx in idxs]
    
    def get_tr_batch_data(self, batch_idx, batch_size):
        batch_indices = slice(batch_idx * batch_size, (batch_idx + 1) * batch_size)

        batch_cov_trajs = self.covariate_trajectories_tr[batch_indices]
        batch_traj_lengths = self.trajectory_lengths_tr[batch_indices]
        batch_cov_trajs = batch_cov_trajs.permute(1, 0, 2)
        batch_packed_cov_trajs = torch.nn.utils.rnn.pack_padded_sequence(
            batch_cov_trajs, batch_traj_lengths, enforce_sorted=False
        )
        batch_event_times = self.event_times_tr[batch_indices]
        batch_censoring_indicators = self.censoring_indicators_tr[batch_indices]
        batch_unshuffle_idxs = self.unshuffled_tr_idxs[batch_indices]
        batch_cov_times = self.cov_times_tr[batch_indices]
        batch_static_covs = self.static_covs_tr[batch_indices]
        batch_missing_indicators = self.missing_indicators_tr[batch_indices]
        return \
            batch_packed_cov_trajs, batch_cov_times, batch_event_times,\
            batch_censoring_indicators, batch_traj_lengths, batch_unshuffle_idxs,\
            batch_static_covs, batch_missing_indicators
            

    def get_tr_data_as_single_batch(self):
        # code uses batches of data
        # so just wrapping tr data in a batch
        # for use in evaluation
        tr_batch = Batch(
            torch.nn.utils.rnn.pack_padded_sequence(
                self.covariate_trajectories_tr.permute(1, 0, 2), 
                self.trajectory_lengths_tr,
                enforce_sorted=False
            ),
            self.cov_times_tr, self.event_times_tr, self.censoring_indicators_tr, 
            self.trajectory_lengths_tr, torch.arange(self.event_times_tr.shape[0]),
            self.static_covs_tr, self.missing_indicators_tr, 
            int(self.max_len_trajectory)
        )
        return tr_batch

    def get_te_data_as_single_batch(self):
        # code uses batches of data
        # so just wrapping te data in a batch
        # for use in evaluation
        te_batch = Batch(
            torch.nn.utils.rnn.pack_padded_sequence(
                self.covariate_trajectories_te.permute(1, 0, 2), 
                self.trajectory_lengths_te,
                enforce_sorted=False
            ),
            self.cov_times_te, self.event_times_te, self.censoring_indicators_te, 
            self.trajectory_lengths_te, torch.arange(self.event_times_te.shape[0]),
            self.static_covs_te, self.missing_indicators_te, 
            int(self.max_len_trajectory)
        )
        return te_batch

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
        batch_static_covs, batch_missing_indicators, max_seq_len_all_batches
    ):

        self.packed_cov_trajs = batch_packed_cov_trajs
        self.cov_times = batch_cov_times
        self.event_times = batch_event_times
        self.censoring_indicators = batch_censoring_indicators
        self.trajectory_lengths = torch.tensor(batch_traj_lengths, dtype=torch.float64)
        self.static_covs = batch_static_covs
        self.unshuffled_idxs = batch_unshuffle_idxs
        self.max_seq_len_all_batches = max_seq_len_all_batches
        self.missing_indicators = batch_missing_indicators

    def get_unpacked_padded_cov_trajs(self):
        batch_covs, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            self.packed_cov_trajs, total_length=self.max_seq_len_all_batches
        )

        batch_covs = batch_covs.transpose(0,1)
        return batch_covs

    '''
        Splits the batch into num_bins sub-batches where each sub-batch has data
        with matched number of covariate events 
        (well matched to have num of cov events within each bin)
        before the start time
        Note that this is only setup to initialize the bins with only the fields
        needed to make evaluation work-so needs to be updated if used elsewhere
    '''
    def split_into_binned_groups_by_num_events(self, num_bins, start_time):
        if start_time == 0:
            return [self], ['no_bins']
        # get num_events per person before start time
        bool_events_before_start = self.cov_times <= start_time
        padding_indicators = \
            (batch.cov_times == 0) &\
            torch.cat([\
                torch.zeros(batch.cov_times.shape[0], 1), \
                torch.ones(
                    batch.cov_times.shape[0], batch.cov_times.shape[1] -1
                )
            ], dim=1).bool()
        bool_events_before_start = torch.where(
            padding_indicators,
            torch.zeros(bool_events_before_start.shape), bool_events_before_start    
        )

        num_events = torch.sum(bool_events_before_start, dim=1)
        bins_per_person = qcut(\
            num_events.cpu().detach().numpy(), num_bins
        ).get_values()
        bin_ends_per_person = [b.right for b in bins_per_person]
        bin_ends = np.unique(bin_ends_per_person)
        batches_grouped_by_bin = []
        for b in range(num_bins):
            bool_idxs_in_bin = bin_ends_per_person == bin_ends[b]
            bin_batch = Batch(
                None, self.cov_times[bool_idxs_in_bin], self.event_times[bool_idxs_in_bin], 
                self.censoring_indicators[bool_idxs_in_bin], None, None,
                None, None, None
            )
            batches_grouped_by_bin.append(bin_batch)

        return batches_grouped_by_bin, bin_ends
            
            
        

        
        
        
        
        

        

