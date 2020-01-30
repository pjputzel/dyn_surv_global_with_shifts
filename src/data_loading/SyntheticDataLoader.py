import DataLoaderBase
import pickle

class SyntheticDataLoader(DataLoaderBase):
    
    def __init__(self, data_loader_params):
        self.params = data_loader_params

    def load_data(self):
        event_times_path = data_loader_params['paths'][0]
        censoring_indicators_path = data_loader_params['paths'][1]
        missing_indicators_path = data_loader_params['paths'][2]
        covariate_trajectories_path = data_loader_params['paths'][3]

        return load_single_path(event_times_path), load_single_path(censoring_indicators_path), load_single_path(missing_indicators_path), load_single_path(covariate_trajectories_path)

    def load_single_path(self, path):
        with open(path, 'rb') as f:
            loaded_object = pickle.load(f)
        return loaded_object 
