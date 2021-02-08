from main_types.LearnFixedThetaBasicMain import LearnFixedThetaBasicMain
import os
import torch
import numpy as np
import pickle
from copy import deepcopy

class BasicMainWithoutSaving(LearnFixedThetaBasicMain):

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        data_input = self.load_data()
        self.preprocess_data(data_input)
        model = self.load_model()
        self.results_tracker = self.train_model(model, data_input)
        # evaluate_model should update the results tracker object to
        # include evaluation metrics
        self.evaluate_model(model, data_input, self.results_tracker)
#        self.save_results(results_tracker)
#        self.plot_results(model, data_input, results_tracker)

class BasicValidationMain(LearnFixedThetaBasicMain):

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        hdim = self.params['model_params']['hidden_dim']
        lr = self.params['train_params']['learning_rate']
        dropout = self.params['model_params']['dropout']
        savedir = os.path.join(
            self.params['savedir'], 
            'validation',
            'hdim%d_lr%.6f_dropout%.2f' %(hdim, lr, dropout)
        )
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.params['savedir'] = savedir

        data_input = self.load_data()
        self.preprocess_data(data_input)
        model = self.load_model()
        self.results_tracker = self.train_model(model, data_input)
        # evaluate_model should update the results tracker object to
        # include evaluation metrics
        self.evaluate_model(model, data_input, self.results_tracker)
        self.save_results(self.results_tracker)
#        self.plot_results(model, data_input, results_tracker)

    def save_results(self, results_tracker):
        with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)

        with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(self.params['savedir'], 'params.pkl'), 'wb') as f:
            pickle.dump(self.params, f) 

class ValidateHiddenDimMain:

    def __init__(self, params):
        self.params = params
#        with open(self.params['data_input_params']['saved_dev_val_te_idxs'], 'rb') as f:
#            self.dev_idxs, self.val_idxs, self.te_idxs = pickle.load(f)


    def main(self):
        # remove pre-saved data input and theta here so we always re-run once
        # before validating to ensure we don't use stale data_input/global_theta
        # for a large run
        split = self.params['data_input_params']['data_loading_params']['paths'].split('/')
        data_dir = ''
        for i in range(len(split) - 1):
            data_dir += split[i] + '/'
        self.data_dir = data_dir        
        data_path = os.path.join(data_dir, 'data_input.pkl')
        theta_filename = 'global_theta_%s.pkl' %self.params['train_params']['loss_params']['distribution_type']
        saved_theta_path = os.path.join(data_dir, theta_filename)
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists(saved_theta_path):
            os.remove(saved_theta_path)

        hidden_dims = self.params['model_params']['hidden_dims_to_validate']

        best_trunc_c_index = -1
        best_hidden_dim_trunc = hidden_dims[0]

        best_standard_c_index = -1
        best_hidden_dim_standard = hidden_dims[0]

        best_standard_trunc_c_index = -1
        best_hidden_dim_standard_trunc = hidden_dims[0]
        for hidden_dim in hidden_dims:
            cur_val_params = deepcopy(self.params)
            cur_val_params['model_params']['hidden_dim'] = hidden_dim
            # change to run the desired main type instead of validation
            cur_val_params['main_type'] = 'learn_fixed_theta_basic_main'
            # update so that we evaluate on the val data instead of test
            cur_val_params['data_input_params']['saved_tr_te_idxs'] = \
                self.params['data_input_params']['saved_dev_val_idxs']

            basic_main = BasicValidationMain(cur_val_params)
            basic_main.main()

            avg_trunc_c_index = \
                torch.mean(basic_main.results_tracker.eval_metrics['c_index_truncated_at_S']['te']['values'][:, 0])
            avg_standard_c_index = \
                torch.mean(basic_main.results_tracker.eval_metrics['c_index']['te']['values'])
            avg_standard_trunc_c_index = \
                torch.mean(basic_main.results_tracker.eval_metrics['standard_c_index_truncated_at_S']['te']['values'])

            if avg_trunc_c_index > best_trunc_c_index:
                best_trunc_c_index = avg_trunc_c_index
                best_hidden_dim_trunc = hidden_dim

            if avg_standard_c_index > best_standard_c_index:
                best_standard_c_index = avg_standard_c_index
                best_hidden_dim_standard = hidden_dim
            
            if avg_standard_trunc_c_index > best_standard_trunc_c_index:
                best_standard_trunc_c_index = avg_standard_trunc_c_index
                best_hidden_dim_standard_trunc = hidden_dim
        print('Best trunc c-index and hidden dim', best_trunc_c_index, best_hidden_dim_trunc)    
        print('Best standard c-index and hidden dim', best_standard_c_index, best_hidden_dim_standard)    
        print('Best standard trunc at S c-index and hidden dim', best_standard_trunc_c_index, best_hidden_dim_standard_trunc)
        # we're now setting up a new main with the winning param training on all tr data 
        # (not just dev)
        # and then evaluating it on the test data instead of the val data
        # so note that the params file must have 
        # both saved_tr_te_idxs and saved_dev_val_idxs with
        # dev val being a split within the tr

        #best_hidden_dims = [best_hidden_dim_standard, best_hidden_dim_standard_trunc, best_hidden_dim_trunc]
        #params_for_final_run = deepcopy(self.params)
        #params_for_final_run['main_type'] = 'learn_fixed_theta_basic_main'
        #params_for_final_run['model_params']['hidden_dim'] = best_hidden_dim
        #basic_main = LearnFixedThetaBasicMain(params_for_final_run)
        #basic_main.main()
