import pandas as pd
import pickle
import numpy as np
import os
import tqdm
import warnings
warnings.filterwarnings("ignore")


FMT_STR = '%d%b%Y:%H:%M:%S'
PATH_PRE = 'Current_output_files_20200723/'
PATH_SUF = '_20200723.csv'
path_func = lambda x: os.path.join(PATH_PRE, x + PATH_SUF)
PATH_DICT = \
    {
        'hospitalized': path_func('hospitalization'),
        'o2': path_func('oxygen'),
        'temp': path_func('temperature'),
        'encs': path_func('ecounters'),
        'meds': path_func('medications'),
        'icu_stay': path_func('icu'),
        'covid_dx': path_func('covid_data_source '),
        'ventilation': path_func('mech_vent'),
        'other_dx': path_func('diagnosis'),
        'labs': path_func('labs'),
        'bp': os.path.join(PATH_PRE, 'fs_bloodpressure_20200731.csv'),
        'demographics': path_func('demographics'),
        'diagnosis': path_func('diagnosis'),
        'NYU_preprocessed': os.path.join(PATH_PRE, 'covid20200726.csv'),
        #'severity': None
    }

def main():
    debug = False
    include_after_hosp_data = True
    time_res_in_days = 1.

    data_type = 'time_to_severe_outcome' #time_to_severe_outcome

    if data_type == 'time_to_death':
        data_pre = COVID19DeathPreprocessor(
            time_res_in_days=time_res_in_days,
            debug=debug, 
        )
    elif data_type == 'time_to_severe_outcome':
        data_pre = COVID19SevereOutcomePreprocessor(
            time_res_in_days=time_res_in_days,
            debug=debug
        )
        
    data_pre.preprocess_data()
    print(data_pre.dynamic_covs_order)
    if not debug:
        with open('new_processed_covid_data.pkl', 'wb') as f:
            pickle.dump(data_pre, f)
    else:
        with open('processed_covid_data_debug.pkl', 'wb') as f:
            pickle.dump(data_pre, f)

#    savename = 'processed_covid_data' + '_debug=%s' %debug + '_afterhosp=%s' %include_after_hosp_data + '.pkl'
#    with open(savename, 'wb') as f:
#        pickle.dump(data_pre, f)

### Variables I wish to use and put together into a single dataframe
### o2 level (nan/-1 if not measured)
### ICU status 0/1 as dynamic value 
### all dynamic measurements from ventilators if on ventilator (-1 if not on ventilator
### temp
### diagnoses at start of the study as a static variable from NRU preprocessed
### demographics as static variables
### blood pressure
### all numeric lab tests
### meds as dynamic categorical variable (same kind of processing as before)
### any other dynamic values from the icu stay data


def convert_cat_vals_to_enumeration_multiple_columns(data_frame, column_names, replace_nans_with=-1):
        enu_to_name = {}
        for column_name in column_names:
            enu_to_name_col = convert_categorical_values_to_enumeration(data_frame, column_name, replace_nans_with=replace_nans_with)
            enu_to_name.update(enu_to_name_col)
        return enu_to_name

def convert_categorical_values_to_enumeration(data_frame, column_name, replace_nans_with=-1):
     unique_values = data_frame[column_name].unique()
     if replace_nans_with == 0:
         enumeration_to_name = {i + 1:med for i, med in enumerate(unique_values)}
     else:
         enumeration_to_name = {i:med for i, med in enumerate(unique_values)}
     enumeration_to_name[replace_nans_with] = 'no value (nan)'    
 
     for i, val in enumerate(unique_values):
         if type(val) == str: 
             if replace_nans_with == 0:
                 data_frame.loc[data_frame[column_name] == val, column_name] = i + 1
             else:
                 data_frame.loc[data_frame[column_name] == val, column_name] = i
         elif not np.isnan(val):
             if replace_nans_with == 0:
                 data_frame.loc[data_frame[column_name] == val, column_name] = i + 1
             else:
                 data_frame.loc[data_frame[column_name] == val, column_name] = i
         
     data_frame.loc[data_frame[column_name].isna(), column_name] = replace_nans_with
     
     return enumeration_to_name


        
#    data_pre.load_unlinked_dfs()
#    data_pre.link_dfs_by_patient_id()
#    data_pre.filter_data_by_hospitalization_status()
#    data_pre.convert_categorical_covariates_to_enumeration()
#    data_pre.combine_all_measurements_per_encounter()
#    data_pre.align_times_by_hospitalization_event()
#    data_pre.split_data()


class COVID19Preprocessor:
    #TODO: all shared default behaviors go here
    # defualt event type can be death
    # default start/end of sequence should be covid dx and event_time (death)
    def __init__(self, 
        path_dict=None, 
        time_res_in_days=.25,
        labtest_missingness_thresh=.25,
        debug=False
    ):
        self.path_dict = PATH_DICT
        if path_dict:
            self.path_dict = path_dict
#        self.event_type = event_type
        self.time_res_in_days = time_res_in_days
        self.labtest_missingness_thresh = labtest_missingness_thresh
        self.debug = debug


    def preprocess_data(self):
        print('Loading CSVs')
        self.load_unlinked_dfs() #
        print('CSVs loaded!')
        self.compute_cohort()
        self.compute_hosp_and_discharge_times()
        self.compute_synch_times()
        self.compute_time_to_events_and_censoring_over_cohort() 
        print('Cohort, time_to_events, and censoring computed!')
        print('Processing dynamic covs...')
        self.compute_dynamic_covs_and_missingness()
        print('Dynamic covs processed!')
#        self.synchronize_event_times_by_hospitalization()
        self.synch_times_to_event()
        self.compute_static_covs()
        print('All Done :)')

        
        
    def load_unlinked_dfs(self):
        for k, v in self.path_dict.items():
            if not self.debug:
                self.__setattr__(k + '_df', pd.read_csv(v))
            else:
                self.__setattr__(k + '_df', pd.read_csv(v.split('.')[0] + '_mini.csv'))

    def compute_cohort(self):
        df = self.NYU_preprocessed_df
        self.cohort_idxs = df['PAT_ID']

    def compute_hosp_and_discharge_times(self):
        df = self.NYU_preprocessed_df
        hospdt_fmt = '%d%b%Y'
        # may have to handle nan case
        convert_to_datetime = \
            lambda x: pd.to_datetime(x, format=hospdt_fmt).astype(np.datetime64)
        convert_to_timedelta = \
            lambda x: np.timedelta64('NaT', 'D') \
            if pd.isna(x) else np.timedelta64(int(x), 'D')
        discharge_times = (
            convert_to_datetime(df['hospdt']) + \
            df['length_stay'].apply(convert_to_timedelta)
        ).values
        
        self.hosp_times = df['hospdt'].astype(np.datetime64).values
        self.discharge_times = discharge_times

    def compute_synch_times(self):
        df = self.NYU_preprocessed_df
        self.synch_times = df['dxdt'].astype(np.datetime64).values
    


    def compute_time_to_events_and_censoring_over_cohort(self):
        df = self.NYU_preprocessed_df.set_index('PAT_ID')
        censoring_indicators = \
            ~df.loc[self.cohort_idxs]['death'].values.astype(bool)
        censoring_indicators = censoring_indicators.astype(int)

        time_to_events = df.loc[self.cohort_idxs][
            'deathdt'
        ].astype(np.datetime64).values

        self.time_to_events = time_to_events
        self.censoring_indicators = censoring_indicators



    def compute_dynamic_covs_and_missingness(self):
        self.drop_labtests_with_high_missingness()
        # just using the Thera Class for now
        print('Number of med types:' , self.meds_df['THERA_CLASS'].unique().shape)
        self.compute_enumeration_of_dynamic_categorical_covs()
        self.convert_measurement_times_to_datetimes()

        dynamic_covs = []
        missing_indicators = []
        meas_times = []
        start_times = []
        end_times = []
        iterate_over = enumerate(
            tqdm.tqdm(self.cohort_idxs)
        )
        pats_to_delete = []
        for i, pat in iterate_over:
            # This needs to be truncated at the time of event or censoring time
            # just stop the time bin iteration at one step before the bin which
            # time to event or censoring falls into
            dynamic_covs_i, missing_i, meas_times_i, start_i, end_i = \
                self.preprocess_encounters_patient_i(i)
            if len(dynamic_covs_i) == 0:
                # the event happened in the same timebin as the first measurement
                pats_to_delete.append(pat)
                continue
            dynamic_covs.append(dynamic_covs_i)
            missing_indicators.append(missing_i)
            meas_times.append(meas_times_i)
            start_times.append(start_i)
            end_times.append(end_i)
        print(pats_to_delete)
        self.remove_pats(pats_to_delete)
        self.dynamic_covs = dynamic_covs
        self.missing_indicators = missing_indicators
        self.meas_times = meas_times 
        self.start_times = start_times
        self.end_times = end_times
        print(len(self.end_times), len(self.cohort_idxs))

    def drop_labtests_with_high_missingness(self):
        # first drop non-numeric labtests
        # then make a pivot table over the component names and result dates
        # aggregating by averaging
        # then get the missingness with the count function
        # then only select those columns with low missingess
        self.labs_df = self.labs_df[
            self.labs_df['PAT_ID'].isin(self.cohort_idxs) &\
            ~(self.labs_df['ORD_NUM_VALUE'] == 9999999)
        ]

        pivot_labs = self.labs_df.pivot_table(
            index='RESULT_DATE', columns='COMPONENT_NAME',
            values='ORD_NUM_VALUE', aggfunc='mean'
        )

        counts = pivot_labs.count()
        print('Only keeping labtests which occur in more than %.3f percent of encounters which is %d encounters' %(self.labtest_missingness_thresh, int(self.labtest_missingness_thresh * len(pivot_labs))))
        labs_to_keep = counts[
            (counts > self.labtest_missingness_thresh * len(pivot_labs))
        ].reset_index()['COMPONENT_NAME'].unique()

        print('Keeping %d labtests' %len(labs_to_keep))
#        self.labs_df = self.labs_df.drop(columns=labs_to_drop)       
        self.labs_df = self.labs_df[
            self.labs_df['COMPONENT_NAME'].isin(labs_to_keep)
        ]

        # not going to work because the result dates and the discharge times have different lengths
        # looks like I was trying to compute missingness of the labs after leaving?
        # or maybe creating a field of labs_after_leaving?
        #labs_after_leaving = self.labs_df[(self.labs_df['PAT_ID'].isin(self.pats_with_labs_after_leaving)) & (self.labs_df['RESULT_DATE'] > self.discharge_times)]
        

    def drop_medications_with_high_missingness(self):
        pass
        #pivot_meds = self.meds_df.pivot_table(index='' 
    def compute_enumeration_of_dynamic_categorical_covs(self):
        ##### O2 #####
        self.o2_device_types_df, self.o2_enu_to_name = \
            self.replace_o2_device_names_with_enumeration()
        

        ##### For now ignoring mech vent features  #####
        ##### because there's a lot of missingness #####
        
        #### For diagnoses I'm thinking of just treating these as a static ####
        #### variable for now-ie what diagnosis is present at covid startup. ####
#        self.diagnosis_df, self.diagnosis_enu_to_name = \
#            convert_categorical_values_to_enumeration(\
#                self.diagnosis_df.set_index('pat_id')[
#                    self.cohort_idxs
#                ].reset_index(),
#                'dx_name'
#            )
        
        #### Note I had to delete a single line (855564) out of the medications ###
        #### csv file that was corrupted somehow ####
        self.meds_enu_to_name = \
            convert_categorical_values_to_enumeration(
                self.meds_df, 'THERA_CLASS' #instead of PHARM_CLASS
            )
        
        # using thera class for now, can also try pharm_class
        self.meds_df = self.meds_df[
            ['PAT_ID', 'PAT_ENC_CSN_ID', 'THERA_CLASS', \
            'START_DATE', 'END_DATE']
        ]

        
        self.labs_enu_to_name = \
            convert_categorical_values_to_enumeration(
                self.labs_df, 'COMPONENT_NAME'
            )
        self.labs_df = self.labs_df[
            ['PAT_ID', 'RESULT_DATE', 'COMPONENT_NAME', 'ORD_VALUE', 'ORD_NUM_VALUE']
        ]

        #self.ordered_dynamic_measurements = ordered_dynamic_measurements
        

    def replace_o2_device_names_with_enumeration(self):
        cohort_o2 = self.o2_df.set_index('PAT_ID').loc[
            self.cohort_idxs
        ].reset_index()
        
        o2_device_types = cohort_o2[
            cohort_o2['FLO_MEAS_NAME'] == 'R OXYGEN DEVICE'
        ]
        
        o2_enu_to_name = convert_categorical_values_to_enumeration(
            o2_device_types, 'MEAS_VALUE'
        )
        return o2_device_types, o2_enu_to_name 



    def preprocess_encounters_patient_i(self, 
        i
    ):
        # note at this point that the encounters should already be correctly 
        # processed so that any 'categorical varibales' (read 'strings') are already
        # filtered
        patient_idx = self.cohort_idxs[i]
        hosp_time = self.hosp_times[i]
        synch_time = self.synch_times[i]
        obs_event_time = self.time_to_events[i]
        measurements_i = self.get_all_encounters_for_patient_i(
            patient_idx, synch_time
        )
        dynamic_covs, missing_indicators, meas_times, start, end = \
            self.collapse_all_measurements_for_patient_i(
                measurements_i, i
            )
        return dynamic_covs, missing_indicators, meas_times, start, end

    def get_all_encounters_for_patient_i(self, patient_idx, synch_time):
        FMT_STR = '%d%b%Y:%H:%M:%S'
        
        o2_meas = self.o2_device_types_df[
            (self.o2_device_types_df['PAT_ID'] == patient_idx) &\
            (self.o2_device_types_df['RECORDED_TIME'].astype(np.datetime64) \
                > synch_time
            )
        ][['RECORDED_TIME', 'MEAS_VALUE']]

        # icu stays
        icu_times = self.icu_stay_df[
            (self.icu_stay_df['PAT_ID'] == patient_idx) &\
            (self.icu_stay_df['DATE'].astype(np.datetime64) \
                > synch_time
            )
        ][['DATE']].astype(np.datetime64)
        icu_times['INDICATOR'] = np.ones(len(icu_times))
        
        temps = self.temp_df[
            (self.temp_df['PAT_ID'] == patient_idx) &\
            (self.temp_df['RECORDED_TIME'].astype(np.datetime64) \
                > synch_time
            )
        ][['RECORDED_TIME', 'MEAS_VALUE']]
         
        # bp
        bp = self.bp_df[
            (self.bp_df['PAT_ID'] == patient_idx) &\
            (self.bp_df['RECORDED_TIME'].astype(np.datetime64) \
                > synch_time
            )
        ][['RECORDED_TIME', 'MEAS_VALUE']]
        def parse_sys_bp(bp_str):
            if type(bp_str) is float:
                return np.nan
            sys_bp = bp_str.split('/')[0]
            return float(sys_bp)
        def parse_dia_bp(bp_str):
            if type(bp_str) is float:
                return np.nan
            dia_bp = bp_str.split('/')[1]
            return float(dia_bp)
        bp['SYS_BP'] = bp['MEAS_VALUE'].apply(parse_sys_bp)
        bp['DIA_BP'] = bp['MEAS_VALUE'].apply(parse_dia_bp)

        labs = self.labs_df[
            (self.labs_df['PAT_ID'] == patient_idx) &\
            (self.labs_df['RESULT_DATE'].astype(np.datetime64) > \
            synch_time)
        ]

        # handles case when end_date is nan (ie medication is currently being handled)
        meds = self.meds_df[
            (self.meds_df['PAT_ID'] == patient_idx) &\
            (
                (self.meds_df['END_DATE'].astype(np.datetime64) > synch_time) |\
                (~self.meds_df['START_DATE'].isna() & self.meds_df['END_DATE'].isna())
            )
        ]

        pat_dfs = {
            'o2': o2_meas,
            'icu_times': icu_times,
            'temp': temps,
            'bp': bp,
            'labs': labs,
            'meds': meds,
        }
        return pat_dfs
         
    

    def convert_measurement_times_to_datetimes(self): 
        self.o2_device_types_df['RECORDED_TIME'] = pd.to_datetime(
            self.o2_device_types_df['RECORDED_TIME'], format=FMT_STR
        ).astype(np.datetime64)

        self.temp_df['RECORDED_TIME'] = pd.to_datetime(
            self.temp_df['RECORDED_TIME'], format=FMT_STR
        ).astype(np.datetime64)

        self.bp_df['RECORDED_TIME'] = pd.to_datetime(
            self.bp_df['RECORDED_TIME'], format=FMT_STR
        ).astype(np.datetime64)

        self.labs_df['RESULT_DATE'] = pd.to_datetime(
            self.labs_df['RESULT_DATE'], format=FMT_STR
        ).astype(np.datetime64)
        self.labs_df['RECORDED_TIME'] = self.labs_df['RESULT_DATE']

        self.meds_df['START_DATE'] = pd.to_datetime(
            self.meds_df['START_DATE'], format=FMT_STR
        ).astype(np.datetime64)

        self.meds_df['END_DATE'] = pd.to_datetime(
            self.meds_df['END_DATE'], format=FMT_STR
        ).astype(np.datetime64)

        FMT_STR2 = '%d%b%Y'
        self.icu_stay_df['DATE'] = pd.to_datetime(
            self.icu_stay_df['DATE'], format=FMT_STR2
        ).astype(np.datetime64)
#        encs_o2 = self.o2_device_types_df[
#            (self.o2_device_types_df['PAT_ID'] == patient_idx) &\
#            (self.o2_device_types_df['RECORDED_TIME'].astype(np.datetime64) \
#                <= hosp_time
#            )
#        ][['EPT_CSN', 'RECORDED_TIME']]
#    
#        encs_icu = self.icu_stay_df[
#            self.icu_stay_df['PAT_ID'] == patient_idx
#            (self.icu_stay_df['DATE'].astype(np.datetime64) \
#                <= hosp_time
#            )
#        ][['PAT_ENC_CSN_ID', 'DATE']]

    ### NOTE: using negative ones for missingness currently
    def collapse_all_measurements_for_patient_i(self, measurements_i, idx):
        obs_event_time = self.time_to_events[idx] 
        cens_ind = self.censoring_indicators[idx]   
        discharge_time = self.discharge_times[idx]
        hosp_time = self.hosp_times[idx]

#        start_day = self.hosp_times[idx].astype(np.datetime64)
        start_day = self.synch_times[idx].astype(np.datetime64)
        time_res_in_hours = np.timedelta64(int(self.time_res_in_days * 24), 'h')
        timebinning = lambda x: (x - start_day)//time_res_in_hours
        for key, val in measurements_i.items():
            df = measurements_i[key]
            if key == 'meds':
                continue
            elif key == 'labs':
                time_bins = df['RESULT_DATE'].apply(timebinning)
            elif key == 'icu_times':
                time_bins = df['DATE'].apply(timebinning)
            else:
                time_bins = df['RECORDED_TIME'].apply(timebinning)
            df['TIME_BIN'] = time_bins
            measurements_i[key] = df
        measurements_i['labs'] = measurements_i['labs'].set_index('TIME_BIN')
        labs_df = measurements_i['labs'].pivot_table(
            index='TIME_BIN', columns='COMPONENT_NAME',
            values='ORD_NUM_VALUE', aggfunc='mean'
        )
        end_day = self.get_end_date(measurements_i, idx)
        num_bins = timebinning(end_day) + 1
        # TODO group rows by bins and collect into a matrix of num_bins by 
        # num total dynamic measurements
        # also create missingness indicators
        collapsed_measurements = []
        missing_indicators = []
        meas_times = []
        for time_bin in range(num_bins):
            # Check if obs_event_time is in the time bin
            end_of_bin = (time_bin + 1) * time_res_in_hours + start_day
            start_of_bin = (time_bin) * time_res_in_hours + start_day
            # end the time bin before the event happens for uncensored individuals
            # otherwise just get all the measurements
            end_of_seq =  ((start_of_bin <= obs_event_time) and (obs_event_time <= end_of_bin)) or (cens_ind and (time_bin == num_bins - 1))

            if end_of_seq:
                break 

            ### Hospitalization status ###
            hosp_ins = (hosp_time >= start_of_bin) and (hosp_time < end_of_bin)
            disch_ins = (discharge_time >= start_of_bin) and (discharge_time < end_of_bin)
            # With neither hosp nor discharge missing
            hosp_or_discharge_in_bin = \
                (~np.isnat(hosp_time) and ~np.isnat(hosp_time))\
                and (hosp_ins or disch_ins)
                
            neither_in_bin_but_overlapping =\
                (~np.isnat(hosp_time) and ~np.isnat(hosp_time))\
                and ((hosp_time < start_of_bin) and (discharge_time >= end_of_bin))
                
        
            # With one of hosp/discharge missing
            # this should only happen when the patient is still hospitalized at the end of study
            if (np.isnat(hosp_time) and not np.isnat(discharge_time)):
                # this shouldn't happen
                print('Patient %s detected to have missing hospitalization time but present discharge time!' %self.cohort_idxs[idx])
            discharge_miss_hosp_before = \
                (np.isnat(discharge_time) and not np.isnat(hosp_time))\
                and (hosp_time <= end_of_bin)

            in_hosp = [int(\
                hosp_or_discharge_in_bin \
                or neither_in_bin_but_overlapping \
                or discharge_miss_hosp_before
            )]
            in_hosp_missingness = [0] # considering as always observed

            ### MEDS ###
            meds_df = measurements_i['meds']
            cond_both_present = \
                (~meds_df['START_DATE'].isna() & ~meds_df['END_DATE'].isna()) &\
                (meds_df['START_DATE'] <= end_of_bin) &\
                (meds_df['END_DATE'] > start_of_bin)
            cond_start_missing = \
                (meds_df['START_DATE'].isna() & ~meds_df['END_DATE'].isna()) &\
                (
                    ((meds_df['END_DATE'] <= end_of_bin) &\
                    (meds_df['END_DATE'] > start_of_bin)) |\
                    (meds_df['END_DATE'] > end_of_bin)
                )    
            cond_end_missing = \
                (~meds_df['START_DATE'].isna() & meds_df['END_DATE'].isna()) &\
                (meds_df['START_DATE'] <= end_of_bin)
            
            # pretty sure not needed                
#            cond_both_missing = \ 
#                (meds_df['START_DATE'],isna() & meds_df['END_DATE'].isna()) &\
                
            # old version without handling the edge cases               
#            meds_at_time = meds_df[
#                (meds_df['START_DATE'] <= end_of_bin) &
#                (meds_df['END_DATE'] > start_of_bin)
#            ]['THERA_CLASS'].unique().astype(str) #or PHARM_CLASS for more detail

            meds_at_time = meds_df[
                cond_both_present | cond_start_missing | cond_end_missing
            ]['THERA_CLASS'].unique().astype(str)

#            meds_missing_at_time = meds_df[
#                ~meds_df['END_DATE'].isna() &\
#                (meds_df['END_DATE'] > end_of_bin)
#            ]

            meds_in_order = sorted(list(
                np.array(list(self.meds_enu_to_name.keys())).astype(str)
            ), key=int)[1:]

            meds_list_at_time = [
                1 if (val in meds_at_time) else 0 
                for val in meds_in_order if not pd.isna(self.meds_enu_to_name[int(val)])
            ]
#            meds_list_at_time2 = [
#                1 if (val in meds_at_time ) else 0 
#                for val in sorted(list(np.array(list(self.meds_enu_to_name.keys())).astype(str)))
#            ]
            meds_missingness = [0 for _ in meds_in_order if not pd.isna(self.meds_enu_to_name[int(_)])]
#            if len(meds_at_time) >=1 :
#                print(meds_at_time, meds_list_at_time)
#                print(sorted(list(np.array(list(self.meds_enu_to_name.values())).astype(str))))
#                print(sorted(list(np.array(list(self.meds_enu_to_name.keys())).astype(str))))
            
            ### LABS ###
#            labs_df = measurements_i['labs']
#            labs_at_time = labs_df[labs_df['TIME_BIN'] == time_bin]
            # Handling edge case where there are no labtests at a given time
            labs_in_order = sorted(list(self.labs_enu_to_name.keys()))[1:]
            if time_bin in labs_df.index:
                labs_at_time = labs_df.loc[time_bin]

#                labs_at_time = labs_at_time.pivot_table(
#                    index='TIME_BIN', columns='COMPONENT_NAME',
#                    values='ORD_NUM_VALUE', aggfunc='mean'
#                )
                included_labs = labs_at_time.index
#                included_labs = labs_at_time['COMPONENT_NAME'].unique()
                # sliced from [1:] to exclude labtests with nan as their comp name
                labs_list_at_time = [
                    labs_at_time[val] if val in included_labs else -1
                    for val in labs_in_order
                ]
                labs_missingness = [
                        1 if (np.isnan(labs_list_at_time[key]) or 
                        (not (key in included_labs))) else 0
                        for key in labs_in_order
                ]
                
            else:
                labs_list_at_time = [
                    -1 for val in list(self.labs_enu_to_name.keys())[1:]
                ]
                labs_missingness =  [1 for key in list(self.labs_enu_to_name.keys())[1:]]
            # TODO do the same for each of the other df's
            ### O2 ###
            o2 = measurements_i['o2']
            o2_at_time = o2[o2['TIME_BIN'] == time_bin]
            if len(o2_at_time) == 0:
                # missing
                o2_at_time = [-1]
                missing_o2 = [1]
            else:
                o2_at_time = [o2_at_time['MEAS_VALUE'].value_counts().argmax()]
#                print(o2_at_time)
                missing_o2 = [0]
    
            ### ICU Times ###
            icu_times = measurements_i['icu_times']
            icu_at_time = icu_times[icu_times['TIME_BIN'] == time_bin]
            if len(icu_at_time) == 0:
                icu_at_time = [0]
            else:
                icu_at_time = [1]
            missing_icu = [0] # considering this as completely observed

            ### BP ###
            bp = measurements_i['bp']
            bp_at_time = bp[bp['TIME_BIN'] == time_bin]
            if len(bp_at_time) == 0:
                bp_at_time = [-1, -1]
                missing_bp = [1, 1]
            else:
                bp_vals_at_time = bp_at_time[['SYS_BP', 'DIA_BP']].values
                bp_vals_at_time = np.nanmean(bp_vals_at_time, axis=0)
#                if bp_vals_at_time.shape[0] > 1:
#                    bp_vals_at_time = np.nanmean(bp_vals_at_time, axis=0)
#                    print(bp_vals_at_time)
#                print([bp for bp in bp_vals_at_time], bp_vals_at_time, np.nanmean(bp_vals_at_time, axis=0))
                bp_at_time = [float(bp) for bp in bp_vals_at_time]
#                bp_at_time = [float(np.nanmean(bp_vals_at_time))]
                missing_bp = [0, 0]
            
            ### TEMP ###
            temp = measurements_i['temp']
            temp_at_time = temp[temp['TIME_BIN'] == time_bin]
            if len(temp_at_time) == 0:
                temp_at_time = [-1]
                missing_temp = [1]
            else:
                # may need to aggregate this
                temps_at_time = list(temp_at_time['MEAS_VALUE'])
                temp_at_time = [np.nanmean(temps_at_time)]
                missing_temp = [0]


            # combine and append
            meas_t = \
                    labs_list_at_time + bp_at_time + temp_at_time + in_hosp +\
                    meds_list_at_time # + o2_at_time + icu_at_time 
            missing_t =\
                    labs_missingness + missing_bp + missing_temp + in_hosp_missingness +\
                    meds_missingness # + missing_o2 + missing_icu

            if np.sum(labs_missingness + missing_bp + missing_temp) == len(labs_missingness + missing_bp + missing_temp):
                # case where all measurements are missing, note that meds and hosp are considered fully
                # observed so they don't count towards determining an empty day.
                continue
            collapsed_measurements.append(meas_t)
            missing_indicators.append(missing_t)
            meas_times.append(time_bin)

        # just to make testing and plotting easier
        if not len(collapsed_measurements) == 0:
            lab_names_in_order = [self.labs_enu_to_name[key] for key in labs_in_order]
            med_names_in_order = [self.meds_enu_to_name[int(key)] for key in meds_in_order if not pd.isna(self.meds_enu_to_name[int(key)])]
            self.dynamic_covs_order = lab_names_in_order + ['sys_bp', 'dia_bp', 'temp', 'hosp'] + med_names_in_order
            self.lab_names_in_order = lab_names_in_order
            self.med_names_in_order = med_names_in_order
        return collapsed_measurements, missing_indicators, meas_times, start_day, end_day

    def get_end_date(self, measurements_i, idx):
        max_times = []
        for k, df in measurements_i.items():
            if k == 'meds':
                continue
            elif k == 'labs':
                max_time = df['RESULT_DATE'].max()
            elif k == 'icu_times':
                max_time = df['DATE'].max()
            else:
                max_time = df['RECORDED_TIME'].max()
            max_times.append(max_time)
        max_times = [pd.Timestamp(0) if pd.isnull(t) else t for t in max_times]
        max_date = np.datetime64(np.max(max_times))
#        if not self.include_after_hosp_data:
        ### Below is only for the case where we truncate at discharge time ###
        ### Default behavior here is not to truncate ###
#        end_of_stay = self.discharge_times[idx]
#        if max_date > end_of_stay:
#            return np.datetime64(end_of_stay)
        return max_date
    
    def remove_pats(self, pats_to_delete):
        self.hosp_times = self.hosp_times[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]
        self.discharge_times = self.discharge_times[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]
        self.time_to_events = self.time_to_events[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]

        self.censoring_indicators = self.censoring_indicators[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]

        self.synch_times = self.synch_times[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]

        self.cohort_idxs = self.cohort_idxs[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]

#    def synchronize_event_times_by_hospitalization(self):
    def synch_times_to_event(self):
        synched_times_to_event = []
        for p, pat in enumerate(tqdm.tqdm(self.cohort_idxs)):
            time = self.time_to_events[p]
            if np.isnat(time):
                time = self.end_times[p]
#            print(type(time), type(self.hosp_times[p]))
            time_relative_to_synch_event = \
                (time - self.synch_times[p]).item()
            synched_times_to_event.append(time_relative_to_synch_event)
#            self.time_to_events[p] = time_relative_to_hosp
        self.censored_event_times = synched_times_to_event
            
    

    def compute_static_covs(self):
        # note: age will be a static covariate this time since people are only
        # in the hospital for shorter periods of time
        column_names = [\
            'SEX', 'RACE', 'ETHNICITY', 'MARITAL_STATUS',  'TOBACCO_USER', 
            'SMOKING_TOB_USE', 'SMOKELESS_TOB_USE'
        ]
        self.enu_to_name_static_covs = convert_cat_vals_to_enumeration_multiple_columns(
            self.demographics_df, column_names
        )
        
        demos = self.demographics_df
        
        demos['BIRTH_DATE'] = pd.to_datetime(
            demos['BIRTH_DATE'], format=FMT_STR
        ).astype(np.datetime64)
        def parse_height(h):
            if pd.isna(h):
                return 65.0
            feet = h.split('\'')[0]
            inches = h.split(' ')[1].split('"')[0]
            return float(feet) * 12 + float(inches)   
        demos['HEIGHT'] = demos['HEIGHT'].apply(parse_height)
        static_covs = []
        for p, pat in enumerate(self.cohort_idxs):
            demos_p = demos[demos['PAT_ID'] == pat]
#            print(type(self.hosp_times[p]), demos_p['BIRTH_DATE'].values, 'meow')
            demos_p['AGE'] = (self.synch_times[p] - demos_p['BIRTH_DATE'].values[0]).item()/(10**9 * 3600 * 24 * 365)
            static_covs.append(demos_p[
                [
                    'AGE', 'SEX', 'RACE', 'ETHNICITY', 'MARITAL_STATUS', \
                    'HEIGHT', 'WEIGHT_POUNDS', 'BMI', 'TOBACCO_USER', \
                    'SMOKING_TOB_USE', 'SMOKELESS_TOB_USE'
                ]
            ].values)
        self.static_covs = static_covs
    

class COVID19SevereOutcomePreprocessor(COVID19Preprocessor):

    def compute_time_to_events_and_censoring_over_cohort(self):
        df = self.NYU_preprocessed_df
        # TODO: compute tte for severity following notes from NYU email
        df['stroke_post_dx'] = np.where(df.strokedt > df.dxdt, 1, 0)
        df['strokedt_post_dx'] = np.where(df.stroke_post_dx, df.strokedt, np.nan)

        df['severe'] = np.where(
            df.icu | df.ventilator | df.death | df.dialysis | df['stroke_post_dx'],
            1, 0
        )
        severe_outcomes = ['icudt', 'ventdt', 'deathdt', 'dialysisdt', 'strokedt_post_dx']
        df['severedt'] = df[severe_outcomes].astype(np.datetime64).min(axis=1)

        df = df.set_index('PAT_ID')
        censoring_indicators = \
            ~df.loc[self.cohort_idxs]['severe'].values.astype(bool)
        censoring_indicators = censoring_indicators.astype(int)

        time_to_events = df.loc[self.cohort_idxs][
            'severedt'
        ].astype(np.datetime64).values

        self.time_to_events = time_to_events
        self.censoring_indicators = censoring_indicators
        

### TODO: update to only change neeeded functions from the base class
class COVID19DeathPreprocessor(COVID19Preprocessor):

    def __init__(self, 
        event_type='DEATH',
        path_dict=None, 
        time_res_in_days=.25,
        labtest_missingness_thresh=.25,
#        include_after_hosp_data=False,
        debug=False
    ):
        self.path_dict = PATH_DICT
        if path_dict:
            self.path_dict = path_dict
        self.event_type = event_type
        if not self.event_type == 'DEATH':
            raise ValueError('Event type %s not yet implemented' %event_type)

        self.time_res_in_days = time_res_in_days
        self.labtest_missingness_thresh = labtest_missingness_thresh
#        self.include_after_hosp_data = include_after_hosp_data
        self.debug = debug
        #self.load_unlinked_dfs()


    def preprocess_data(self):
        print('Loading CSVs')
        self.load_unlinked_dfs() #
        print('CSVs loaded!')
        # for case of using data after leaving the hospital, just leaving this func the same for now
        # may update later
        self.compute_hospitalization_times_and_cohort() #potentially change the discharge times to the final time of censoring, and then everything should be same? Otherwise you could just not change the discharge times and add if else statements to handle the patients after leaving with a flag
        self.compute_time_to_events_and_censoring_over_cohort() # death times for patients in the hospital is all that is computed for now I'm pretty sure, but we need to check this
        print('Cohort, time_to_events, and censoring computed!')
        print('Processing dynamic covs...')
        self.compute_dynamic_covs_and_missingness()
        print('Dynamic covs processed!')
        self.synchronize_event_times_by_hospitalization()
        self.compute_static_covs()
        print('All Done :)')

        
        
    def load_unlinked_dfs(self):
        for k, v in self.path_dict.items():
            if not self.debug:
                self.__setattr__(k + '_df', pd.read_csv(v))
            else:
                self.__setattr__(k + '_df', pd.read_csv(v.split('.')[0] + '_mini.csv'))


    def compute_hospitalization_times_and_cohort(self):
        df = self.NYU_preprocessed_df
        hosp_covid_df = df[~df['hospdt'].isna()]
        hosp_pat_idxs = hosp_covid_df['PAT_ID'].values
        hosp_times = hosp_covid_df['hospdt'].astype(np.datetime64).values
        
        hospdt_fmt = '%d%b%Y'
        hosp_covid_df['hospdt'] = pd.to_datetime(
            hosp_covid_df['hospdt'], format=hospdt_fmt
        ).astype(np.datetime64)
        convert_to_timedelta = \
            lambda x: np.timedelta64('NaT', 'D') \
            if pd.isna(x) else np.timedelta64(int(x), 'D')
        discharge_times = (
            hosp_covid_df['hospdt'] + \
            hosp_covid_df['length_stay'].apply(convert_to_timedelta)
        ).values

        
        
        self.cohort_idxs = hosp_pat_idxs
        self.hosp_times = hosp_times
        self.discharge_times = discharge_times

        # check how much data we have after discharge
        #labs_after_leaving = self.labs_df[
        #    self.labs_df['PAT_ID'].isin(self.cohort_idxs) &\
        #    (self.labs_df['RESULT_DATE'] > self.discharge_times)
        #]
        self.labs_df['RESULT_DATE'] = pd.to_datetime(
            self.labs_df['RESULT_DATE'], format=FMT_STR
        ).astype(np.datetime64)
        labs_df = self.labs_df[self.labs_df['PAT_ID'].isin(self.cohort_idxs)].set_index('PAT_ID').loc[self.cohort_idxs].reset_index()
        max_lab_meas_df = labs_df.groupby('PAT_ID').max().reset_index()[['PAT_ID', 'RESULT_DATE']]
        labs_after_leaving = max_lab_meas_df[max_lab_meas_df['RESULT_DATE'] > self.discharge_times]
        deaths_after_leaving = max_lab_meas_df[
            (max_lab_meas_df['RESULT_DATE'] > self.discharge_times) &\
            ~(self.demographics_df['DEATH_DATE'].isna())
        ]
        self.pats_with_labs_after_leaving = labs_after_leaving['PAT_ID'].unique()

        print('Num with labs after leaving', len(labs_after_leaving))
        print('Num Deaths after leaving hospital', len(deaths_after_leaving))

    def compute_time_to_events_and_censoring_over_cohort(self):
#        if not self.include_after_hosp_data:
        df = self.NYU_preprocessed_df.set_index('PAT_ID')
        censoring_indicators = \
            ~df.loc[self.cohort_idxs]['death'].values.astype(bool)
        censoring_indicators = censoring_indicators.astype(int)

        time_to_events = df.loc[self.cohort_idxs][
            'deathdt'
        ].astype(np.datetime64).values
#        else:
#            df = self.demographics_df.set_index('pat_id')
#            censoring_indicators = \
#                ~df.loc[self.cohort_idxs]['DEATH_DATE'].isna().astype(int)
#            time_to_events = df.loc[self.cohort_idxs][
#                'DEATH_DATE'
#            ].astype(np.datetime64).values

        self.time_to_events = time_to_events
        self.censoring_indicators = censoring_indicators



    def compute_dynamic_covs_and_missingness(self):
        self.drop_labtests_with_high_missingness()
        # just using the Thera Class for now
        #self.drop_rare_medications()
        print('Number of med types:' , self.meds_df['THERA_CLASS'].unique().shape)
        self.compute_enumeration_of_dynamic_categorical_covs()
        self.convert_measurement_times_to_datetimes()

        dynamic_covs = []
        missing_indicators = []
        meas_times = []
        start_times = []
        end_times = []
        iterate_over = enumerate(zip(
            tqdm.tqdm(self.cohort_idxs),
            self.hosp_times
        ))
        pats_to_delete = []
        for i, (pat, time) in iterate_over:
            # This needs to be truncated at the time of event or censoring time
            # just stop the time bin iteration at one step before the bin which
            # time to event or censoring falls into
            dynamic_covs_i, missing_i, meas_times_i, start_i, end_i = \
                self.preprocess_encounters_patient_i(i)
            if len(dynamic_covs_i) == 0:
                # the event happened in the same timebin as the first measurement
                pats_to_delete.append(pat)
                continue
            dynamic_covs.append(dynamic_covs_i)
            missing_indicators.append(missing_i)
            meas_times.append(meas_times_i)
            start_times.append(start_i)
            end_times.append(end_i)
#        np.delete(
#            self.cohort_idxs,
#            np.in1d(self.cohort_idxs, pats_to_delete)
#        )
        self.remove_pats(pats_to_delete)
        self.dynamic_covs = dynamic_covs
        self.missing_indicators = missing_indicators
        self.meas_times = meas_times 
        self.start_times = start_times
        self.end_times = end_times
        print(len(self.end_times), len(self.cohort_idxs))

    def drop_labtests_with_high_missingness(self):
        # first drop non-numeric labtests
        # then make a pivot table over the component names and result dates
        # aggregating by averaging
        # then get the missingness with the count function
        # then only select those columns with low missingess
        self.labs_df = self.labs_df[
            self.labs_df['PAT_ID'].isin(self.cohort_idxs) &\
            ~(self.labs_df['ORD_NUM_VALUE'] == 9999999)
        ]

        pivot_labs = self.labs_df.pivot_table(
            index='RESULT_DATE', columns='COMPONENT_NAME',
            values='ORD_NUM_VALUE', aggfunc='mean'
        )

        counts = pivot_labs.count()
        print('Only keeping labtests which occur in more than %.3f percent of encounters which is %d encounters' %(self.labtest_missingness_thresh, int(self.labtest_missingness_thresh * len(pivot_labs))))
        labs_to_keep = counts[
            (counts > self.labtest_missingness_thresh * len(pivot_labs))
        ].reset_index()['COMPONENT_NAME'].unique()

        print('Keeping %d labtests' %len(labs_to_keep))
#        self.labs_df = self.labs_df.drop(columns=labs_to_drop)       
        self.labs_df = self.labs_df[
            self.labs_df['COMPONENT_NAME'].isin(labs_to_keep)
        ]

        # not going to work because the result dates and the discharge times have different lengths
        # looks like I was trying to compute missingness of the labs after leaving?
        # or maybe creating a field of labs_after_leaving?
        labs_after_leaving = self.labs_df[(self.labs_df['PAT_ID'].isin(self.pats_with_labs_after_leaving)) & (self.labs_df['RESULT_DATE'] > self.discharge_times)]
        

    def drop_medications_with_high_missingness(self):
        pass
        #pivot_meds = self.meds_df.pivot_table(index='' 
    def compute_enumeration_of_dynamic_categorical_covs(self):
        ### TODO: compute union of all dynamic covs and medications 
        ### Note that categorical values are represented as a vector of
        ### ones and zeros of length equal to the number of categories and
        ### one representing the presence of that category at a given encounter
        ### ordering is arbitrary, but needs to be fixed of course.

        # get union of all desired dynamic covs
        # these are as follows:
        ### o2 device used (nan/-1 if not measured, categorical for which type of
        ###     device was used)
        ### ICU status 0/1 as dynamic value - compute in per pat traj func
        ### all dynamic measurements from ventilators if on ventilator (-1 if not on ventilator
        ### temp not categorical
        ### diagnoses at start of the study as a static variable from NRU preprocessed handled in the staic cov function
        ### demographics as static variables (handled in static covariate function)
        ### blood pressure (handled in per pat traj func)
        ### all numeric lab tests 
        ### meds as dynamic categorical variable (same kind of processing as before)
        ### any other dynamic values from the icu stay data
         ### based on nyu preprocessed df:
         ### before/after dialyis treatment
         ### before/after stroke

        ### Other stuff
        ### 1. handle missingness for medications in the same way as before (in pat traj loop)
        ### 2. before/after dialysis as a variable  (in pat traj loop)
        ### 3. before/after stroke (in pat traj loop)
        ##### O2 #####
        self.o2_device_types_df, self.o2_enu_to_name = \
            self.replace_o2_device_names_with_enumeration()
        

        ##### For now ignoring mech vent features  #####
        ##### because there's a lot of missingness #####
        
        #### For diagnoses I'm thinking of just treating these as a static ####
        #### variable for now-ie what diagnosis is present at covid startup. ####
#        self.diagnosis_df, self.diagnosis_enu_to_name = \
#            convert_categorical_values_to_enumeration(\
#                self.diagnosis_df.set_index('PAT_ID')[
#                    self.cohort_idxs
#                ].reset_index(),
#                'DX_NAME'
#            )
        
        #### Note I had to delete a single line (855564) out of the medications ###
        #### csv file that was corrupted somehow ####
        self.meds_enu_to_name = \
            convert_categorical_values_to_enumeration(
                self.meds_df, 'THERA_CLASS' #instead of PHARM_CLASS
            )
        
        # using thera class for now, can also try pharm_class
        self.meds_df = self.meds_df[
            ['PAT_ID', 'PAT_ENC_CSN_ID', 'THERA_CLASS', \
            'START_DATE', 'END_DATE']
        ]

        
        self.labs_enu_to_name = \
            convert_categorical_values_to_enumeration(
                self.labs_df, 'COMPONENT_NAME'
            )
        self.labs_df = self.labs_df[
            ['PAT_ID', 'RESULT_DATE', 'COMPONENT_NAME', 'ORD_VALUE', 'ORD_NUM_VALUE']
        ]

        #self.ordered_dynamic_measurements = ordered_dynamic_measurements
        

    def replace_o2_device_names_with_enumeration(self):
        cohort_o2 = self.o2_df.set_index('PAT_ID').loc[
            self.cohort_idxs
        ].reset_index()
        
        o2_device_types = cohort_o2[
            cohort_o2['FLO_MEAS_NAME'] == 'R OXYGEN DEVICE'
        ]
        
        o2_enu_to_name = convert_categorical_values_to_enumeration(
            o2_device_types, 'MEAS_VALUE'
        )
        return o2_device_types, o2_enu_to_name 



    def preprocess_encounters_patient_i(self, 
        i
    ):
        # to do this then get the csn's from the encounters file for an individual
        # note the encounter file start_date is at the resolution of days
        # while the actual hours/minutes for each different type of measurement
        # will be correctly listed in a separate file for that measurement
        # note at this point that the encounters should already be correctly 
        # processed so that any 'categorical varibales' (read 'strings') are already
        # filtered
        patient_idx = self.cohort_idxs[i]
        hosp_time = self.hosp_times[i]
        obs_event_time = self.time_to_events[i]
        measurements_i = self.get_all_encounters_for_patient_i(
            patient_idx, hosp_time
        )
        dynamic_covs, missing_indicators, meas_times, start, end = \
            self.collapse_all_measurements_for_patient_i(
                measurements_i, i
            )
        return dynamic_covs, missing_indicators, meas_times, start, end

    def get_all_encounters_for_patient_i(self, patient_idx, hosp_time):
        FMT_STR = '%d%b%Y:%H:%M:%S'
        
        o2_meas = self.o2_device_types_df[
            (self.o2_device_types_df['PAT_ID'] == patient_idx) &\
            (self.o2_device_types_df['RECORDED_TIME'].astype(np.datetime64) \
                > hosp_time
            )
        ][['RECORDED_TIME', 'MEAS_VALUE']]

        # icu stays
        icu_times = self.icu_stay_df[
            (self.icu_stay_df['PAT_ID'] == patient_idx) &\
            (self.icu_stay_df['DATE'].astype(np.datetime64) \
                > hosp_time
            )
        ][['DATE']].astype(np.datetime64)
        icu_times['INDICATOR'] = np.ones(len(icu_times))
        
        temps = self.temp_df[
            (self.temp_df['PAT_ID'] == patient_idx) &\
            (self.temp_df['RECORDED_TIME'].astype(np.datetime64) \
                > hosp_time
            )
        ][['RECORDED_TIME', 'MEAS_VALUE']]
         
        # bp
        bp = self.bp_df[
            (self.bp_df['PAT_ID'] == patient_idx) &\
            (self.bp_df['RECORDED_TIME'].astype(np.datetime64) \
                > hosp_time
            )
        ][['RECORDED_TIME', 'MEAS_VALUE']]
        def parse_sys_bp(bp_str):
            if type(bp_str) is float:
                return np.nan
            sys_bp = bp_str.split('/')[0]
            return float(sys_bp)
        def parse_dia_bp(bp_str):
            if type(bp_str) is float:
                return np.nan
            dia_bp = bp_str.split('/')[1]
            return float(dia_bp)
        bp['SYS_BP'] = bp['MEAS_VALUE'].apply(parse_sys_bp)
        bp['DIA_BP'] = bp['MEAS_VALUE'].apply(parse_dia_bp)

        labs = self.labs_df[
            (self.labs_df['PAT_ID'] == patient_idx) &\
            (self.labs_df['RESULT_DATE'].astype(np.datetime64) > \
            hosp_time)
        ]

        meds = self.meds_df[
            (self.meds_df['PAT_ID'] == patient_idx) &\
            (self.meds_df['END_DATE'].astype(np.datetime64) >\
             hosp_time)
            
        ]

        pat_dfs = {
            'o2': o2_meas,
            'icu_times': icu_times,
            'temp': temps,
            'bp': bp,
            'labs': labs,
            'meds': meds,
        }
        return pat_dfs
         
    

    def convert_measurement_times_to_datetimes(self): 
        self.o2_device_types_df['RECORDED_TIME'] = pd.to_datetime(
            self.o2_device_types_df['RECORDED_TIME'], format=FMT_STR
        ).astype(np.datetime64)

        self.temp_df['RECORDED_TIME'] = pd.to_datetime(
            self.temp_df['RECORDED_TIME'], format=FMT_STR
        ).astype(np.datetime64)

        self.bp_df['RECORDED_TIME'] = pd.to_datetime(
            self.bp_df['RECORDED_TIME'], format=FMT_STR
        ).astype(np.datetime64)

        self.labs_df['RESULT_DATE'] = pd.to_datetime(
            self.labs_df['RESULT_DATE'], format=FMT_STR
        ).astype(np.datetime64)
        self.labs_df['RECORDED_TIME'] = self.labs_df['RESULT_DATE']

        self.meds_df['START_DATE'] = pd.to_datetime(
            self.meds_df['START_DATE'], format=FMT_STR
        ).astype(np.datetime64)

        self.meds_df['END_DATE'] = pd.to_datetime(
            self.meds_df['END_DATE'], format=FMT_STR
        ).astype(np.datetime64)

        FMT_STR2 = '%d%b%Y'
        self.icu_stay_df['DATE'] = pd.to_datetime(
            self.icu_stay_df['DATE'], format=FMT_STR2
        ).astype(np.datetime64)
#        encs_o2 = self.o2_device_types_df[
#            (self.o2_device_types_df['PAT_ID'] == patient_idx) &\
#            (self.o2_device_types_df['RECORDED_TIME'].astype(np.datetime64) \
#                <= hosp_time
#            )
#        ][['EPT_CSN', 'RECORDED_TIME']]
#    
#        encs_icu = self.icu_stay_df[
#            self.icu_stay_df['PAT_ID'] == patient_idx
#            (self.icu_stay_df['DATE'].astype(np.datetime64) \
#                <= hosp_time
#            )
#        ][['PAT_ENC_CSN_ID', 'DATE']]

    ### NOTE: using negative ones for missingness currently
    def collapse_all_measurements_for_patient_i(self, measurements_i, idx):
        obs_event_time = self.time_to_events[idx] 
        cens_ind = self.censoring_indicators[idx]   
        discharge_time = self.discharge_times[idx]

        start_day = self.hosp_times[idx].astype(np.datetime64)
        time_res_in_hours = np.timedelta64(int(self.time_res_in_days * 24), 'h')
        timebinning = lambda x: (x - start_day)//time_res_in_hours
        for key, val in measurements_i.items():
            df = measurements_i[key]
            if key == 'meds':
                continue
            elif key == 'labs':
                time_bins = df['RESULT_DATE'].apply(timebinning)
            elif key == 'icu_times':
                time_bins = df['DATE'].apply(timebinning)
            else:
                time_bins = df['RECORDED_TIME'].apply(timebinning)
            df['TIME_BIN'] = time_bins
            measurements_i[key] = df
        measurements_i['labs'] = measurements_i['labs'].set_index('TIME_BIN')
        labs_df = measurements_i['labs'].pivot_table(
            index='TIME_BIN', columns='COMPONENT_NAME',
            values='ORD_NUM_VALUE', aggfunc='mean'
        )
        end_day = self.get_end_date(measurements_i, idx)
        num_bins = timebinning(end_day) + 1
        # TODO group rows by bins and collect into a matrix of num_bins by 
        # num total dynamic measurements
        # also create missingness indicators
        collapsed_measurements = []
        missing_indicators = []
        meas_times = []
        for time_bin in range(num_bins):
            # Check if obs_event_time is in the time bin
            end_of_bin = (time_bin + 1) * time_res_in_hours + start_day
            start_of_bin = (time_bin) * time_res_in_hours + start_day
            # end the time bin before the event happens for uncensored individuals
            # otherwise just get all the measurements
            end_of_seq =  (start_of_bin <= obs_event_time) and (obs_event_time <= end_of_bin) or (cens_ind and (time_bin == num_bins - 1))
            if end_of_seq:
                break 

            ### Hospitalization status ###
            in_hosp = int(\
                not ( (discharge_time >= start_of_bin)  and (discharge_time <= end_of_bin) )
            )

            ### MEDS ###
            meds_df = measurements_i['meds']
            meds_at_time = meds_df[
                (meds_df['START_DATE'] <= end_of_bin) &
                (meds_df['END_DATE'] > start_of_bin)
            ]['THERA_CLASS'].unique().astype(str) #or PHARM_CLASS for more detail
            meds_list_at_time = [
                1 if (val in meds_at_time) else 0 
                for val in sorted(list(np.array(list(self.meds_enu_to_name.keys())).astype(str)), key=int)[1:] if not pd.isna(self.meds_enu_to_name[int(val)])
            ]
#            meds_list_at_time2 = [
#                1 if (val in meds_at_time ) else 0 
#                for val in sorted(list(np.array(list(self.meds_enu_to_name.keys())).astype(str)))
#            ]
            meds_missingness = [0 for _ in sorted(list(np.array(list(self.meds_enu_to_name.keys())).astype(str)), key=int)[1:]  if not pd.isna(self.meds_enu_to_name[int(_)])]
#            if len(meds_at_time) >=1 :
#                print(meds_at_time, meds_list_at_time)
#                print(sorted(list(np.array(list(self.meds_enu_to_name.values())).astype(str))))
#                print(sorted(list(np.array(list(self.meds_enu_to_name.keys())).astype(str))))
            
            ### LABS ###
#            labs_df = measurements_i['labs']
#            labs_at_time = labs_df[labs_df['TIME_BIN'] == time_bin]
            # Handling edge case where there are no labtests at a given time
            if time_bin in labs_df.index:
                labs_at_time = labs_df.loc[time_bin]

#                labs_at_time = labs_at_time.pivot_table(
#                    index='TIME_BIN', columns='COMPONENT_NAME',
#                    values='ORD_NUM_VALUE', aggfunc='mean'
#                )
                included_labs = labs_at_time.index
#                included_labs = labs_at_time['COMPONENT_NAME'].unique()
                # sliced from [1:] to exclude labtests with nan as their comp name
                labs_list_at_time = [
                    labs_at_time[val] if val in included_labs else -1
                    for val in sorted(list(self.labs_enu_to_name.keys()))[1:]
                ]
                labs_missingness = [
                        1 if (np.isnan(labs_list_at_time[key]) or 
                        (not (key in included_labs))) else 0
                        for key in sorted(list(self.labs_enu_to_name.keys()))[1:]
                ]
                
            else:
                labs_list_at_time = [
                    -1 for val in list(self.labs_enu_to_name.keys())[1:]
                ]
                labs_missingness =  [1 for key in list(self.labs_enu_to_name.keys())[1:]]
            # TODO do the same for each of the other df's
            ### O2 ###
            o2 = measurements_i['o2']
            o2_at_time = o2[o2['TIME_BIN'] == time_bin]
            if len(o2_at_time) == 0:
                # missing
                o2_at_time = [-1]
                missing_o2 = [1]
            else:
                o2_at_time = [o2_at_time['MEAS_VALUE'].value_counts().argmax()]
#                print(o2_at_time)
                missing_o2 = [0]
    
            ### ICU Times ###
            icu_times = measurements_i['icu_times']
            icu_at_time = icu_times[icu_times['TIME_BIN'] == time_bin]
            if len(icu_at_time) == 0:
                icu_at_time = [0]
            else:
                icu_at_time = [1]
            missing_icu = [0] # considering this as completely observed

            ### BP ###
            bp = measurements_i['bp']
            bp_at_time = bp[bp['TIME_BIN'] == time_bin]
            if len(bp_at_time) == 0:
                bp_at_time = [-1, -1]
                missing_bp = [1, 1]
            else:
                bp_vals_at_time = bp_at_time[['SYS_BP', 'DIA_BP']].values
                bp_vals_at_time = np.nanmean(bp_vals_at_time, axis=0)
#                if bp_vals_at_time.shape[0] > 1:
#                    bp_vals_at_time = np.nanmean(bp_vals_at_time, axis=0)
#                    print(bp_vals_at_time)
#                print([bp for bp in bp_vals_at_time], bp_vals_at_time, np.nanmean(bp_vals_at_time, axis=0))
                bp_at_time = [float(bp) for bp in bp_vals_at_time]
#                bp_at_time = [float(np.nanmean(bp_vals_at_time))]
                missing_bp = [0, 0]
            
            ### TEMP ###
            temp = measurements_i['temp']
            temp_at_time = temp[temp['TIME_BIN'] == time_bin]
            if len(temp_at_time) == 0:
                temp_at_time = [-1]
                missing_temp = [1]
            else:
                # may need to aggregate this
                temps_at_time = list(temp_at_time['MEAS_VALUE'])
                temp_at_time = [np.nanmean(temps_at_time)]
                missing_temp = [0]

            
#            ### Hospitalization ###
#            hosp = 


            # combine and append
#            print(type(labs_list_at_time), type(meds_list_at_time), type(o2), type(icu_at_time), type(bp_at_time), type(temp_at_time))
            meas_t = \
                    labs_list_at_time + bp_at_time + temp_at_time + \
                    meds_list_at_time + o2_at_time + icu_at_time 
            missing_t =\
                    labs_missingness + missing_bp + missing_temp + \
                    meds_missingness + missing_o2 + missing_icu
            collapsed_measurements.append(meas_t)
            missing_indicators.append(missing_t)
            meas_times.append(time_bin)
        return collapsed_measurements, missing_indicators, meas_times, start_day, end_day

    def get_end_date(self, measurements_i, idx):
        max_times = []
        for k, df in measurements_i.items():
            if k == 'meds':
                continue
            elif k == 'labs':
                max_time = df['RESULT_DATE'].max()
            elif k == 'icu_times':
                max_time = df['DATE'].max()
            else:
                max_time = df['RECORDED_TIME'].max()
            max_times.append(max_time)
        max_times = [pd.Timestamp(0) if pd.isnull(t) else t for t in max_times]
        max_date = np.datetime64(np.max(max_times))
#        if not self.include_after_hosp_data:
        end_of_stay = self.discharge_times[idx]
        if max_date > end_of_stay:
            return np.datetime64(end_of_stay)
        return max_date
    
    def remove_pats(self, pats_to_delete):
        self.hosp_times = self.hosp_times[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]
        self.discharge_times = self.discharge_times[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]
        self.time_to_events = self.time_to_events[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]

        self.censoring_indicators = self.censoring_indicators[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]
        self.cohort_idxs = self.cohort_idxs[\
            ~np.in1d(self.cohort_idxs, pats_to_delete)
        ]

    def synchronize_event_times_by_hospitalization(self):
        synched_time_to_events = []
        for p, pat in enumerate(tqdm.tqdm(self.cohort_idxs)):
            time = self.time_to_events[p]
            if np.isnat(time):
                time = self.end_times[p]
#            print(type(time), type(self.hosp_times[p]))
            time_relative_to_hosp = \
                (time - self.hosp_times[p]).item()
            synched_time_to_events.append(time_relative_to_hosp)
#            self.time_to_events[p] = time_relative_to_hosp
        self.censored_event_times = synched_time_to_events
            
    

    def compute_static_covs(self):
        # note: age will be a static covariate this time since people are only
        # in the hospital for shorter periods of time
        column_names = [\
            'SEX', 'RACE', 'ETHNICITY', 'MARITAL_STATUS',  'TOBACCO_USER', 
            'SMOKING_TOB_USE', 'SMOKELESS_TOB_USE'
        ]
        self.enu_to_name_static_covs = convert_cat_vals_to_enumeration_multiple_columns(
            self.demographics_df, column_names
        )
        
        demos = self.demographics_df
        
        demos['BIRTH_DATE'] = pd.to_datetime(
            demos['BIRTH_DATE'], format=FMT_STR
        ).astype(np.datetime64)
        def parse_height(h):
            if pd.isna(h):
                return 65.0
            feet = h.split('\'')[0]
            inches = h.split(' ')[1].split('"')[0]
            return float(feet) * 12 + float(inches)   
        demos['HEIGHT'] = demos['HEIGHT'].apply(parse_height)
        static_covs = []
        for p, pat in enumerate(self.cohort_idxs):
            demos_p = demos[demos['PAT_ID'] == pat]
#            print(type(self.hosp_times[p]), demos_p['BIRTH_DATE'].values, 'meow')
            demos_p['AGE'] = (self.hosp_times[p] - demos_p['BIRTH_DATE'].values[0]).item()/(10**9 * 3600 * 24 * 365)
            static_covs.append(demos_p[
                [
                    'AGE', 'SEX', 'RACE', 'ETHNICITY', 'MARITAL_STATUS', \
                    'HEIGHT', 'WEIGHT_POUNDS', 'BMI', 'TOBACCO_USER', \
                    'SMOKING_TOB_USE', 'SMOKELESS_TOB_USE'
                ]
            ].values)
        self.static_covs = static_covs
#### Old Comments
    ### Old comments for collapse function
        # here we collapse with the time resolution we decide on 
        # so we take the df made in get_all_encounters_for_patient_i and
        # we bin it into the nearest self.time_res_in_days interval per encoutner
        # adding that as an extra field. Then we create a column for each 
        # dynamic measurement and for every encounter in the same bin we pull
        # the values from the correct dataframe and insert them into the columns
        # so we'll iterate over all the times in the discretization for each
        # individual
    ### Old comments for the get_all_encounters_patient_i function
        # for every type of dynamic measurement, pull all the encounter csn's
        # as well as the encounter times and return a df (with 3 columns)
        # with the csn's, the times, and the file name the csn comes
        # fromas the columns

        ### o2 device used (nan/-1 if not measured, categorical for which type of
        ###     device was used)
        ### ICU status 0/1 as dynamic value - compute in per pat traj func
        ### all dynamic measurements from ventilators if on ventilator (-1 if not on ventilator
        ### temp
        ### diagnoses at start of the study as a static variable from NRU preprocessed
        ### demographics as static variables
        ### blood pressure
        ### all numeric lab tests
        ### meds as dynamic categorical variable (same kind of processing as before)
        ### any other dynamic values from the icu stay data
if __name__ == '__main__':
    main()
