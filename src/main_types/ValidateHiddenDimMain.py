from main_types.LearnFixedThetaBasicMain import LearnFixedThetaBasicMain
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

class BasicMainWithoutPlotting(LearnFixedThetaBasicMain):

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
        self.save_results(results_tracker)
#        self.plot_results(model, data_input, results_tracker)

    def save_results(self, results_tracker):
        #TODO implement saving here such that the te_preds are saved, the model is saved,
        # and the specified evaluation metrics are saved.

        # Directory should be saved based on the selected hyperparameters
        raise NotImplementedError('Need to implement correct saving')

class ValidateHiddenDimMain:

    def __init__(self, params):
        self.params = params
#        with open(self.params['data_input_params']['saved_dev_val_te_idxs'], 'rb') as f:
#            self.dev_idxs, self.val_idxs, self.te_idxs = pickle.load(f)


    def main(self):
        hidden_dims = self.params['model_params']['hidden_dims_to_validate']

        best_c_index = -1
        best_hidden_dim = hidden_dims[0]
        for hidden_dim in hidden_dims:
            cur_val_params = deepcopy(self.params)
            cur_val_params['model_params']['hidden_dim'] = hidden_dim
            # change to run the desired main type instead of validation
            cur_val_params['main_type'] = 'learn_fixed_theta_basic_main'
            # update so that we evaluate on the val data instead of test
            cur_val_params['data_input_params']['saved_tr_te_idxs'] = \
                self.params['data_input_params']['saved_dev_val_idxs']

            basic_main = BasicMainWithoutPlotting(cur_val_params)
            basic_main.main()

            avg_trunc_c_index = \
                torch.mean(basic_main.results_tracker.eval_metrics['c_index_truncated_at_S']['te']['values'][:, 0])
            if avg_trunc_c_index > best_c_index:
                best_c_index = avg_trunc_c_index
                best_hidden_dim = hidden_dim
        print('Best c-index and hidden dim', best_c_index, best_hidden_dim)    
        # we're now setting up a new main with the winning param training on all tr data 
        # (not just dev)
        # and then evaluating it on the test data instead of the val data
        # so note that the params file must have 
        # both saved_tr_te_idxs and saved_dev_val_idxs with
        # dev val being a split within the tr
        params_for_final_run = deepcopy(self.params)
        params_for_final_run['main_type'] = 'learn_fixed_theta_basic_main'
        params_for_final_run['model_params']['hidden_dim'] = best_hidden_dim
        basic_main = LearnFixedThetaBasicMain(params_for_final_run)
        basic_main.main()
