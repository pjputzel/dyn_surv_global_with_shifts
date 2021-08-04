from data_handling.DataLoaderBase import DataLoaderBase
import numpy as np
import pandas as pd
import pickle

DISC_STATIC_COV_IDXS = [0, 1]
NUM_CATEGORIES_DISC_STATIC = {
    0:2, 1:2
}


class PBC2DataLoader(DataLoaderBase):

    def __init__(self, data_loader_params):
        self.params = data_loader_params

    def load_data(self):
        data_path = self.params['paths']
        df = pd.read_csv(data_path)
        # drop rows where the end of sequence is the same as the last day of
        # measurement
        df = df[~(df['times'] == df['tte'])] 
        # code later own assumes categories start from 0, and histologic starts
        # from one, so have to subtract one
        df['histologic'] = df['histologic'] - 1
        means_by_ids = df.groupby('id').mean() 
        event_times = means_by_ids['tte'].values
        censoring_indicators = (~means_by_ids['label'].values.astype(bool)).astype(int)
        censoring_indicators[means_by_ids['label'].values == 2] = 1
        static_covs = means_by_ids[['drug', 'sex', 'age']].values
        
        ind_idxs = df.id.unique()
        trajs = []
        missing_inds = []
        dynamic_covs = [\
            'ascites', 'hepatomegaly', 'spiders', 'edema', 
            'serBilir', 'serChol', 'albumin', 'alkaline', 
            'SGOT', 'platelets', 'prothrombin', 'histologic'
        ]
        for ind_idx in ind_idxs:
            ind_df = df[df.id == ind_idx]
            times = ind_df['times']
            traj_ind = []
            missing_ind = []
            for time in times:
                df_at_time = ind_df[df.times == time]
                
                missing_at_time = np.where(
                    np.isnan(df_at_time[dynamic_covs].values),
                    np.ones(len(dynamic_covs)),
                    np.zeros(len(dynamic_covs))
                )

                traj_ind.append(
                    [time, list(df_at_time[dynamic_covs].values.squeeze())]
                )
                missing_ind.append(list(missing_at_time.squeeze()))
            trajs.append(traj_ind)
            missing_inds.append(missing_ind) 
        if self.params['one_hot_encode_static_vars']:
            static_vars =\
                self.convert_static_vars_to_bit_strings_with_missingness(static_covs)
        if self.params['one_hot_encode_dynamic_disc_vars']:
            trajs, missing_inds =\
                self.convert_dynamic_discrete_vars_to_bit_strings(trajs, missing_inds)
        print('Length of dynamic covs: %s, length of static covs: %s' %(len(trajs[0][0][1]), len(static_vars[0])))
        missing_inds = [[[float(entry) for entry in m] for m in missingness_i] for missingness_i in missing_inds]
        replace_nans_with = -1
        
        norm = self.params['timescale']
        rescale_func = lambda x: x/norm
        
        trajs = [
            [
                [rescale_func(values[0]), values[1]] 
                for j, values in enumerate(list(cov_values_i))
            ]
            for i, cov_values_i in enumerate(trajs)
        ]
        event_times = [rescale_func(event_time) for event_time in event_times]
        return event_times, censoring_indicators, missing_inds, trajs, static_vars

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
    def convert_dynamic_discrete_vars_to_bit_strings(self, cov_trajs, missing_inds):
        '''Converts the cov_trajs discrete values to bit strings (one-hot). If miss
        -ing then every entry in the bit string will be zero, and missing_inds will
        contain a single one.
        '''
        # maps each index of disc_var to it's number of categories
        disc_vars_info = {
            0:2, 1:2, 2:2, 3:3, len(cov_trajs[0][0][1]) - 1 : 4
        }
        def one_hot_encode_single_cov_vector(entry, missing_inds):
            time, entry = entry[0], entry[1]
            disc_vars = []
            for disc_var_idx in disc_vars_info.keys():
                num_cats = disc_vars_info[disc_var_idx]
                bit_vec = [0 for i in range(num_cats)]
                val = entry[disc_var_idx] 

                # vals range from 0 -> num_categories - 1
                # vector will be of length num_categories
                # zero for example, will map to (0*)1
                
                if missing_inds[disc_var_idx] == 0:
                    bit_vec[int(val)] = 1
                disc_vars = disc_vars + bit_vec
            cont_vals = []
            cont_miss = []
            disc_miss = []
            start_idx = 0
            for disc_var_idx in disc_vars_info.keys():
                # build up continuous values in the entry before discrete vals
                # for both missing_inds and the entry
                cont_vals.extend(entry[start_idx:disc_var_idx])
                cont_miss.extend(missing_inds[start_idx:disc_var_idx])
                disc_miss.append(missing_inds[disc_var_idx])
                start_idx = disc_var_idx + 1 
            # add the last section after the last discrete var idx
            if not start_idx == len(entry): 
                # case where last idx isn't discrete
                cont_vals.extend(entry[start_idx:])
                cont_miss.extend(missing_inds[start_idx:])
                
            
            # add discrete at the end for both missing_inds and the entry
            one_hot_entry = cont_vals + disc_vars
            missing_for_entry = cont_miss + disc_miss
            return [time, one_hot_entry], missing_for_entry

        trajs_with_one_hot = []
        missing_order_updated = []
        for i, traj_ind in enumerate(cov_trajs):
            traj_ind_with_one_hot = []
            missing_ind = []
            for j, entry in enumerate(traj_ind):
                one_hot_entry, matching_missing = one_hot_encode_single_cov_vector(entry, missing_inds[i][j])
                traj_ind_with_one_hot.append(one_hot_entry)
                missing_ind.append(matching_missing)
            trajs_with_one_hot.append(traj_ind_with_one_hot)
            missing_order_updated.append(missing_ind)
        return trajs_with_one_hot, missing_order_updated

