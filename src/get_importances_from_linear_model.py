import sys
sys.path.append('../data/COVID-19')
from preprocess_data import COVID19SevereOutcomePreprocessor
import numpy as np
import torch
import pickle
import pandas as pd

NUM_CATEGORIES_DISC_STATIC = { 
    1:2, 2:9, 3:4, 4:9, 8:6, 9:11, 10:4
}

NUM_NOT_DISC_PRIOR = [1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4]
def get_missingness_static_feat(i, static_covs):
#    if i in NUM_CATEGORIES_DISC_STATIC:
#        miss_idx = np.sum([
#            NUM_CATEGORIES_DISC_STATIC[j] + 1 for j in NUM_CATEGORIES_DISC_STATIC.keys() if j < i
#        ])
#        miss_idx += NUM_NOT_DISC_PRIOR[i] + NUM_CATEGORIES_DISC_STATIC[i]
#        miss_idx = int(miss_idx)
#        miss_percent = 1/(len(static_covs)) * np.sum([
#            [static_cov[miss_idx] for static_cov in static_covs_i]
#            for static_covs_i in static_covs
#        ])
#    else:
#        miss_percent = 0
    miss_percent =  1/(len(static_covs)) * np.sum([((static_covs_i[0][i] == -1) or (np.isnan(static_covs_i[0][i]))) for static_covs_i in static_covs])
    if i in NUM_CATEGORIES_DISC_STATIC:
        miss_percents = [miss_percent for k in range(NUM_CATEGORIES_DISC_STATIC[i] + 1)] 
    else:
        miss_percents = [miss_percent]
    return miss_percents
#    return miss_percent

if __name__ == '__main__':
    with open('../output/covid/rayleigh/linear_delta_per_step/model.pkl', 'rb') as f:
        model = pickle.load(f)

    
    with open('../data/COVID-19/severity_processed_covid_data.pkl', 'rb') as f:
        data = pickle.load(f)
    static_covs_ori = \
        [
            'AGE', 'SEX', 'RACE', 'ETHNICITY', 'MARITAL_STATUS', \
            'HEIGHT', 'WEIGHT_POUNDS', 'BMI', 'TOBACCO_USER', \
            'SMOKING_TOB_USE', 'SMOKELESS_TOB_USE' 
        ]
    # Add in categories
    temp = []
    for c, cov in enumerate(static_covs_ori):
        if c in NUM_CATEGORIES_DISC_STATIC:
            temp.extend([cov + '_cat%d' %i for i in range(NUM_CATEGORIES_DISC_STATIC[c])])
            #temp.extend([cov for i in range(NUM_CATEGORIES_DISC_STATIC[c])])
            temp.append(cov + '_missing')
        else:
            temp.append(cov)
    static_covs = temp
    print(len(static_covs))
    print(len(data.static_covs[0][0]))

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
    missing_percents_static = [get_missingness_static_feat(i, data.static_covs) for i in range(len(static_covs_ori))]
    temp = []
    for percents in missing_percents_static:
        temp = temp + percents
    missing_percents_static = temp
    missing_percents = [0.] + list(missing_percents) + [0. for _ in range(len(missing_percents))] + missing_percents_static
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
    
