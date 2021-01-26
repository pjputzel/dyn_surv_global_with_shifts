import os
from main_types.BasicMain import BasicMain
import pickle
import torch
import numpy as np
from loss.LossCalculator import LossCalculator

class SaveLearnedStandardRisksMain:

    def __init__(self, params):
        self.basic_main = BasicMain(params)
        self.params = params    

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        data_input = self.basic_main.load_data()
        self.basic_main.preprocess_data(data_input)
        model = self.load_model()
        model.eval()
        loss = LossCalculator(self.params['train_params']['loss_params'], self.params['model_params']['model_type'] )
        data = data_input.get_te_data_as_single_batch()
        pred_params, _, _ = model(data)

        start_times = self.params['eval_params']['dynamic_metrics']['start_times']
        window_max_length = self.params['eval_params']['dynamic_metrics']['window_length']
        window_size = self.params['eval_params']['dynamic_metrics']['time_step']
        windows = \
            [(i + 1) * window_size for i in range(window_max_length//window_size)]

        savedir = os.path.join(self.basic_main.params['savedir'], 'standard_c_index_risks')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        for start_time in start_times:
            for window in windows:
                savepath = os.path.join(savedir, 'risks_time%d_window%d.csv' %(start_time, window))
                risks = loss.logprob_calculator.compute_most_recent_CDF(
                    pred_params, data, model.get_global_param(),
                    start_time, window
                )
                np.savetxt(savepath, risks.detach().numpy(), delimiter=',')

    def load_model(self):
        model_path = os.path.join(
            self.basic_main.params['savedir'],
            'model.pkl'
        )
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

