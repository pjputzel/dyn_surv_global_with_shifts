import numpy as np
import pickle
from utils.ParameterParser import ParameterParser
from main_types.BasicMain import BasicMain
from data_handling.COVIDSevereOutcomePreprocessor import COVID19SevereOutcomePreprocessor
from evaluation.ModelEvaluator import ModelEvaluator
import torch

def compute_deephit_risks_ik(pred_time, preds, data):
    # person X 1 X time_bins
    sort_idxs = np.argsort(data.event_times)
    sorted_event_times = data.event_times[sort_idxs]
    sorted_cens_ind = data.censoring_indicators[sort_idxs]
    sorted_preds = preds[sort_idxs]

    normalization = 1. - np.sum(sorted_preds[:, 0:pred_time + 1], axis=1).squeeze()
#    normalization = normalization
    #print(normalization)
    risks_ik = []
    for i, time in enumerate(sorted_event_times):
        risks_k = 1/(normalization[i]) * np.sum(sorted_preds[:, pred_time + 1:int(time) + 1], axis=1).squeeze()
        risks_ik.append(risks_k)
    risks_ik = np.array(risks_ik)
    #print(risks_ik.shape, normalization.shape, sorted_preds.shape)
#    print(np.sum(sorted_preds, axis=1) )
    #print(risks_ik)
#    unc_at_risk = \
#        ~sorted_cens_ind.bool() &\
#        (sorted_event_times > pred_time)
#    ret = risks_ik[unc_at_risk]
#    print(ret.shape)
    ret = risks_ik
    return ret


if __name__ == '__main__':

#    path_to_deephit_preds = '../output/covid/preds_deephit_te_default_lr_divided_by_ten_winner.pkl'
#    path_to_deephit_preds = '../output/covid/preds_deephit_val_lr_divided_by_ten_hidden_state_RNN=50.pkl'
    #path_to_deephit_preds = '../output/covid/preds_deephit_val_lr_divided_by_ten_hidden_state_RNN=100.pkl'
#    path_to_deephit_preds = '../output/covid/preds_deephit_val_default_everything.pkl'

#    path_to_deephit_preds = '../output/covid/severity_deephit_preds/lr0.001_hdim50_preds_val.pkl'
#    path_to_deephit_preds = '../output/covid/severity_deephit_preds/lr0.001_hdim100_preds_val.pkl'
#    path_to_deephit_preds = '../output/covid/severity_deephit_preds/lr0.001_hdim200_preds_val.pkl'
#    path_to_deephit_preds = '../output/covid/severity_deephit_preds/lr0.00001_hdim200_preds_val.pkl'
    path_to_deephit_preds = '../output/covid/severity_deephit_preds/lr0.001_hdim200_preds_te_winner.pkl'

    #params = '../configs/linear_baseline_configs/covid_configs/linear_delta_per_step_covid.yaml' #just to get the data loaded properly
    params = '../configs/RNN_based_model_configs/covid_configs/learn_fixed_theta_RNN_delta_per_step_covid_severity.yaml' #just to get the data loaded properly
    # load data again
    # then get the full te batch to evaluate
    params = ParameterParser(params).parse_params()
    params['data_input_params']['saved_tr_te_idxs'] = '../data/COVID-19/severity_dev_val_idxs_covid.pkl'
    pred_times = params['eval_params']['dynamic_metrics']['start_times']
    torch.random.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])
    torch.set_default_dtype(torch.float64)
    main = BasicMain(params)
    data = main.load_data().get_te_data_as_single_batch()
        

    
    with open(path_to_deephit_preds, 'rb') as f:
        preds_all = pickle.load(f)
    preds_all = [pred.squeeze() for pred in preds_all]

    evaluator = ModelEvaluator(params['eval_params'], params['train_params']['loss_params'], params['model_params']['model_type'])
    c_indices = []
    for p, preds in enumerate(preds_all):
        risks = compute_deephit_risks_ik(int(pred_times[p]), preds, data)
        total_concordant_pairs, total_valid_pairs = evaluator.compute_c_index_upper_boundary_at_event_time_i(torch.tensor(risks), data, int(pred_times[p]))
        if total_valid_pairs == 0:
            c_index = 0
        else:
            c_index = total_concordant_pairs/total_valid_pairs

        print('Truncated C-index for pred time %d is %.3f' %(int(pred_times[p]), c_index))
        c_indices.append(c_index)

    print('Average at-risk c-index is %.4f' %np.mean(np.array(c_indices)))

    
     
