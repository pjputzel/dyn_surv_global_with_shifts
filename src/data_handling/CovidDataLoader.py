import sys
sys.path.append('/home/pj/Documents/Dynamic SA/DDGGD/DDGGD/src')
from data_handling.DataLoaderBase import DataLoaderBase
import numpy as np
import pandas
import pickle
DEBUG = False


DISC_STATIC_COV_IDXS = [1, 2, 3, 4, 8, 9, 10]
NUM_CATEGORIES_DISC_STATIC = {
    1:2, 2:9, 3:4, 4:9, 8:6, 9:11, 10:4
}
class CovidDataLoader(DataLoaderBase):

    def __init__(self, data_loader_params):
        self.params = data_loader_params
        self.data_path = self.params['paths']
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.o2_enu_to_name = data.o2_enu_to_name

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

    def remove_disc_static_covs(self, static_covs):
        def convert_single_ind(svars):
            temp_svars = []
            for s, svar in enumerate(svars):
                if s in DISC_STATIC_COV_IDXS:
                    continue
                else:
                    svar = [svar]
                temp_svars = temp_svars + svar
            return temp_svars
        static_vars = [
            convert_single_ind(svars)
            for svars in static_covs
        ]
        return static_vars
        

    def load_data(self):
        data = self.data
        event_times = data.censored_event_times
        censoring_indicators = list(data.censoring_indicators)
        dynamic_covs = data.dynamic_covs
        missing_indicators = data.missing_indicators
        static_vars = data.static_covs
        meas_times = data.meas_times
        self.dynamic_covs_order = data.dynamic_covs_order
        if DEBUG:
            idxs = np.random.permutation(np.arange(len(event_times)))[0:50]
            event_times = [event_times[i] for i in idxs]
            censoring_indicators = [censoring_indicators[i] for i in idxs]
            missing_indicators = [missing_indicators[i] for i in idxs]
            dynamic_covs = [dynamic_covs[i] for i in idxs]
            static_vars = [static_vars[i] for i in idxs]
            meas_times = [data.meas_times[i] for i in idxs]

        static_vars = [
            list(static_vars_i[0]) for static_vars_i in static_vars
        ]

        if self.params['one_hot_encode_static_vars']:
            static_vars =\
                self.convert_static_vars_to_bit_strings_with_missingness(static_vars)
        
        print('Length of dynamic covs: %s, length of static covs: %s' %(len(dynamic_covs[0][0]), len(static_vars[0])))
        missing_indicators = [[[float(entry) for entry in m] for m in list(missingness_i)] for missingness_i in missing_indicators]
    
        replace_nans_with = -1
        trajs = [
            [
                [float(meas_times[individual][i]),  [val if not np.isnan(val) else replace_nans_with for val in values]] 
                for i, values in enumerate(list(cov_values_i))
            ]
            for individual, cov_values_i in enumerate(dynamic_covs)
        ]
        norm = (10**9 * 3600 * 24) #normalize from nanoseconds to days
        event_times = [event_time/norm for event_time in event_times]

        return event_times, censoring_indicators, missing_indicators, trajs, static_vars
        
