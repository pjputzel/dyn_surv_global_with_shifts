from data_handling.DataLoaderBase import DataLoaderBase
import numpy as np
import pandas
import pickle

DISC_STATIC_COV_IDXS = [0, 1, 2, 3]
NUM_CATEGORIES_DISC_STATIC = {
    0:2, 1:9, 2:2, 3:3 
}
class DmCvdDataLoader(DataLoaderBase):

    def __init__(self, data_loader_params):
        self.params = data_loader_params

    def convert_static_vars_to_bit_strings_with_missingness(self, static_vars):
        '''Convert all static vars to one hot encoded bit strings with missingness

        Args:
            static_vars (list): Static vars per individual
        
        Returns:
            list: Static vars one hot encoded with missing indicator at the
                end of each expanded bit string for each variable.
            
        '''

        def convert_single_ind(svars):
            temp_svars = []
            for s, svar in enumerate(svars):
                if s in DISC_STATIC_COV_IDXS:
                    # plus one for the missing indicator at the end
                    bit_str = [0 for i in range(NUM_CATEGORIES_DISC_STATIC[s] + 1)]
                    if np.isnan(svar):
                        # it's missing so map to end of bit string
                        svar = -1
                    bit_str[int(svar)] = 1.
                    svar = bit_str
                else:
                    svar = [svar]
                temp_svars = temp_svars + svar
            return temp_svars
        static_vars = [
            convert_single_ind(svars)
            for svars in static_vars
        ]
        return static_vars

    def load_data(self):
        data_path = self.params['paths']
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        event_times = data[0]
        censoring_indicators = data[1]
        trajs = data[2]
        missing_indicators = data[3]
        static_vars = data[4]
        if self.params['one_hot_encode_static_vars']:
            static_vars =\
                self.convert_static_vars_to_bit_strings_with_missingness(static_vars)
        
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
#        norm = 365
        norm = 1
        trajs = [
            [   
                [traj_t[0]/norm, [cov_value for c, cov_value in enumerate(traj_t[1])]]
                for t, traj_t in enumerate(traj)
            ] 
            for i, traj in enumerate(trajs)
        ]
        event_times = [event_time/norm for event_time in event_times]
        return event_times, censoring_indicators, missing_indicators, trajs, static_vars
        
