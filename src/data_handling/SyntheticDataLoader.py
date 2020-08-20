from data_handling.DataLoaderBase import DataLoaderBase
import pickle

class SyntheticDataLoader(DataLoaderBase):
    
    def __init__(self, data_loader_params):
        self.params = data_loader_params

    def load_data(self):
        event_times_path = self.params['paths'][2]
        censoring_indicators_path = self.params['paths'][1]
        covariate_trajectories_path = self.params['paths'][0]
        covariate_trajectories = self.load_single_path(covariate_trajectories_path)
        formatted_trajectories = []
        for traj in covariate_trajectories:
            formatted_trajectories.append([[traj_t[0], [traj_t[1]]] for traj_t in traj])
        # nothing is missing for synth data currently
        # note in general that the missing_indicators will be a nested list with
        # each element having length equal to the covariate dim. Or maybe just a numpy array?
        missing_indicators = [[[0. for i in range(len(traj[j][1]))] for j in range(len(traj))] for traj in formatted_trajectories]
        # no static covs either 
        static_covs = [[0.] for traj in formatted_trajectories]

        return self.load_single_path(event_times_path), self.load_single_path(censoring_indicators_path), missing_indicators, formatted_trajectories, static_covs

    def load_single_path(self, path):
        with open(path, 'rb') as f:
            loaded_object = pickle.load(f)
        return loaded_object 


class SimpleSyntheticDataLoader(DataLoaderBase):

    def __init__(self, data_loader_params):
        self.params = data_loader_params
        
    def load_data(self):
        with open(self.params['paths'], 'rb') as f:
            data = pickle.load(f)

        covariate_trajectories = data.cov_trajs
        formatted_trajectories = []
        for traj in covariate_trajectories:
            formatted_trajectories.append([[traj_t[0], [traj_t[1]]] for traj_t in traj])

        # nothing is missing for synth data currently
        missing_indicators = [[[0. for i in range(len(traj[j][1]))] for j in range(len(traj))] for traj in formatted_trajectories]
       
        # no static covs either 
        static_covs = [[0.] for traj in formatted_trajectories]
        
        # no censoring in simple data
        censoring = [0 for i in range(len(data.true_event_times))]
        return data.true_event_times, censoring, missing_indicators, formatted_trajectories, static_covs
