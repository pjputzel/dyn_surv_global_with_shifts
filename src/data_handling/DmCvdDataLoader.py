from data_handling.DataLoaderBase import DataLoaderBase
import numpy as np
import pandas
import pickle

class DmCvdDataLoader(DataLoaderBase):

    def __init__(self, data_loader_params):
        self.params = data_loader_params

    def load_data(self):
        data_path = self.params['paths']
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        event_times = data[0]
        censoring_indicators = data[1]
        trajs = data[2]
        missing_indicators = data[3]
        static_vars = data[4]
        
        #print([len(m) for m in missing_indicators])
        print('Length of dynamic covs: %s, length of static covs: %s' %(len(trajs[0][0][1]), len(static_vars[0])))
        missing_indicators = [[[float(entry) for entry in m] for m in missingness_i] for missingness_i in missing_indicators]
        
        # lets try masking out values to zero
        # TODO: make this an option-either masking to zero or use averages
#        trajs = [
#            [   
#                [traj_t[0]/365., [cov_value * (1 - missing_indicators[i][t][c]) for c, cov_value in enumerate(traj_t[1])]]
#                for t, traj_t in enumerate(traj)
#            ] 
#            for i, traj in enumerate(trajs)
#        ]
        max_event_time = np.max(np.array(event_times))
#        norm = max_event_time
        norm = 365
        trajs = [
            [   
                [traj_t[0]/norm, [cov_value for c, cov_value in enumerate(traj_t[1])]]
                for t, traj_t in enumerate(traj)
            ] 
            for i, traj in enumerate(trajs)
        ]
        event_times = [event_time/norm for event_time in event_times]
        return event_times, censoring_indicators, missing_indicators, trajs, static_vars
        
