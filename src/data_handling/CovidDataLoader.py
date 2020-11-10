import sys
#sys.path.append('../data/COVID-19/')
#from preprocess_data import COVID19_Preprocessor
#from data_handling.COVID19_Preprocessor import COVID19_Preprocessor
from data_handling.DataLoaderBase import DataLoaderBase
import numpy as np
import pandas
import pickle

class CovidDataLoader(DataLoaderBase):

    def __init__(self, data_loader_params):
        self.params = data_loader_params

    def load_data(self):
        data_path = self.params['paths']
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        event_times = data.censored_event_times
        censoring_indicators = list(data.censoring_indicators)
        dynamic_covs = data.dynamic_covs
        missing_indicators = data.missing_indicators
        static_vars = data.static_covs
        
        static_vars = [
            list(static_vars_i[0]) for static_vars_i in static_vars
        ]
        print(static_vars[0])
        
        print('Length of dynamic covs: %s, length of static covs: %s' %(len(dynamic_covs[0][0]), len(static_vars[0])))
        missing_indicators = [[[float(entry) for entry in m] for m in list(missingness_i)] for missingness_i in missing_indicators]
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
    
        max_event_time = np.max(np.array(event_times))
        median_event_time = np.median(np.array(event_times))
#       norm = max_event_time
#        norm = 365
#        norm = median_event_time
        replace_nans_with = -1
        norm = (10**9 * 3600 * 24) #normalize from nanoseconds to days
        trajs = [
            [
                [float(i) * data.time_res_in_days,  [val if not np.isnan(val) else replace_nans_with for val in values]] # times are discretized here, get real time by multiplying index by the time resolution
                for i, values in enumerate(list(cov_values_i))
            ]
            for cov_values_i in dynamic_covs
        ]
        event_times = [event_time/norm for event_time in event_times]
#        print(len(missing_indicators[0][0]), len(trajs[0][0][1]))
#        trajs = [
#            [   
#                [traj_t[0]/norm, [cov_value for c, cov_value in enumerate(traj_t[1])]]
#                for t, traj_t in enumerate(traj)
#            ] 
#            for i, traj in enumerate(trajs)
#        ]
        return event_times, censoring_indicators, missing_indicators, trajs, static_vars
        
