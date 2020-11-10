import sys
from data_handling.DataLoaderBase import DataLoaderBase
import numpy as np
import pandas
import pickle

class MimicDataLoader(DataLoaderBase):


    def __init__(self, data_loader_params):
        self.params = data_loader_params

    def load_data(self):
        data_path = self.params['paths']
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        event_times = data['cens_event_times']
        censoring_indicators = list(data['cens_ind'])
        dynamic_covs = data['trajs']
        missing_indicators = data['missing_ind']
        
        # only a few static vars, just including them in dynamic
        # just using 0s here
        static_vars = [
            [0.] for idx in data['files']
        ]
        
        print('Length of dynamic covs: %s, length of static covs: %s' %(len(dynamic_covs[0][0]), len(static_vars[0])))
        missing_indicators = [[[float(entry) for entry in m[1:]] for m in list(missingness_i)] for missingness_i in missing_indicators]
        
        # lets try masking out values to zero
        # TODO: make this an option-either masking to zero or use averages
#        trajs = [
#            [   
#                [traj_t[0]/365., [cov_value * (1 - missing_indicators[i][t][c]) for c, cov_value in enumerate(traj_t[1])]]
#                for t, traj_t in enumerate(traj)
#            ] 
#            for i, traj in enumerate(trajs)
#        ]
        
        # fix formatting to traj style used in DMCvd data
    
#        max_event_time = np.max(np.array(event_times))
        median_event_time = np.median(np.array(event_times))
#        norm = max_event_time
#        norm = median_event_time
        norm = 1.
#        norm = 365
        trajs = [
            [
                [values[0]/norm, list(values[1:])]
                for values in list(cov_values_i)
            ]
            for cov_values_i in dynamic_covs
        ]
#        print([len(traj[0][1]) for traj in trajs])
#        print([len(event[1]) for event in trajs[0]])
        event_times = [event_time/norm for event_time in event_times]
#        print(event_times)
#        print([len(missing_indicator[0]) for missing_indicator in missing_indicators])
#        print([len(missing_event) for missing_event in missing_indicators[0]])
#        trajs = [
#            [   
#                [traj_t[0]/norm, [cov_value for c, cov_value in enumerate(traj_t[1])]]
#                for t, traj_t in enumerate(traj)
#            ] 
#            for i, traj in enumerate(trajs)
#        ]
        print(len(trajs[0][0][1]), 'meooww')
        return event_times, censoring_indicators, missing_indicators, trajs, static_vars
        
