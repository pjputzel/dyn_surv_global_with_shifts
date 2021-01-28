import sys
sys.path.append('../data/COVID-19')
from preprocess_data import COVID19SevereOutcomePreprocessor
import numpy as np
import torch
import pickle
import pandas as pd

if __name__ == '__main__':
    with open('../output/covid/rayleigh/linear_delta_per_step/model.pkl', 'rb') as f:
        model = pickle.load(f)

    
    with open('../data/COVID-19/severity_processed_covid_data.pkl', 'rb') as f:
        data = pickle.load(f)
    static_covs = \
        [
            'AGE', 'SEX', 'RACE', 'ETHNICITY', 'MARITAL_STATUS', \
            'HEIGHT', 'WEIGHT_POUNDS', 'BMI', 'TOBACCO_USER', \
            'SMOKING_TOB_USE', 'SMOKELESS_TOB_USE' 
        ]
    missing_names = [name + '_missing' for name in data.dynamic_covs_order]
    var_names = ['time'] + data.dynamic_covs_order +  missing_names + static_covs 
    
    # compute missing per var name (except for time and static covs which aren't missing)
    missing_counts = np.zeros(len(data.dynamic_covs_order))
    tot_encs = 0
    for i, ind_traj in enumerate(data.dynamic_covs):
        for j, enc in enumerate(ind_traj):
            missing_counts = missing_counts + np.array(data.missing_indicators[i][j])
            tot_encs += 1 
    missing_percents = missing_counts/tot_encs 
    missing_percents = [0.] + list(missing_percents) + [0. for _ in range(len(missing_percents))] + [0. for _ in range(len(static_covs))]
    print(missing_percents)
    print(var_names)

    weights = model.linear.weight.detach().numpy()[0]
    idxs = np.argsort(-np.abs(weights))
    sorted_var_names = [var_names[idx] for idx in idxs]
    sorted_weights = weights[idxs]
    sorted_missing_percents = [missing_percents[idx] for idx in idxs]
   
    
 
    df = pd.DataFrame(\
        {
            'Variable Names':sorted_var_names,
            'Weights':sorted_weights,
            'Missing %': sorted_missing_percents
        }
    )
    df = df.set_index('Variable Names')
    df.to_csv('linear_weights_with_names.csv')
    
