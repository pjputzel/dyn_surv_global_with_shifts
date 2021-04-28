from main_types.BasicMain import BasicMain 
from utils.Diagnostics import Diagnostics
import os
import pickle
from evaluation.ModelEvaluator import ModelEvaluator
import torch
import numpy as np
from models.LandmarkedCoxModel import LandmarkedCoxModel
import matplotlib.pyplot as plt

class DiagnoseDependentCensoringMain(BasicMain):

    def __init__(self, params):
        self.params = params
        self.params['savedir'] = os.path.join(\
            params['savedir_pre'], 'diagnose_dependent_censoring'
        )
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])
        self.device = torch.device(self.params['device'])

    def load_model(self):
        # just have start times be a single value which has reasonable
        # performance for the previous results.
        self.model_tte = LandmarkedCoxModel(
            self.params['model_params'], 
            self.params['eval_params']['dynamic_metrics']['start_times']
        )
        self.model_ttc = LandmarkedCoxModel(
            self.params['model_params'], 
            self.params['eval_params']['dynamic_metrics']['start_times']
        )
        return self.model_tte

    def train_model(self, model, data_input):
        # just handling the training here
        for landmark_time in self.params['eval_params']['dynamic_metrics']['start_times']:
            landmark_data = data_input.get_tr_landmarked_dataset(landmark_time)
            self.train_single_cox_model(
                self.model_tte.models[landmark_time],
                landmark_data, verbose=False
            )
        print('Training model on censoring times with flipped indicators')
        self.data_input.switch_censoring_indicators()
        for landmark_time in self.params['eval_params']['dynamic_metrics']['start_times']:
            landmark_data = data_input.get_tr_landmarked_dataset(landmark_time)
            self.train_single_cox_model(
                self.model_ttc.models[landmark_time],
                landmark_data, lr_weight=.1, verbose=False
            )
        # check that this is correct todo when not needing diagnositcs
        self.diagnostics = Diagnostics(self.params['train_params']['diagnostic_params'])
        return self.diagnostics

    def train_single_cox_model(self, model, data, verbose=False, lr_weight=1.):
        event_indicators = (~data['censoring'].astype(bool)).astype(int)
        model.fit(
            data['covs'], data['event_times'], event_indicators,
            max_iter=self.params['train_params']['max_iter'],
            lr=self.params['train_params']['learning_rate'] * lr_weight,
            tol=self.params['train_params']['conv_thresh'],
            l2_reg=self.params['train_params']['loss_params']['cox_l2_reg'],
            verbose=verbose
        )

    
    def evaluate_model(self, model, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            self.params['model_params']['model_type']
        )
        self.model_evaluator.evaluate_model(
            self.model_tte, data_input, diagnostics
        )

    def save_results(self, results_tracker):
        start_time = self.params['eval_params']['dynamic_metrics']['start_times'][0]
        try:
            with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
                pickle.dump(results_tracker, f)
        except:
            print('Tracker save file too large to save!')

        with open(os.path.join(self.params['savedir'], 'model_tte.pkl'), 'wb') as f:
            pickle.dump(self.model_tte.models[start_time], f)
        
        with open(os.path.join(self.params['savedir'], 'model_ttc.pkl'), 'wb') as f:
            pickle.dump(self.model_ttc.models[start_time], f)

        with open(os.path.join(self.params['savedir'], 'params.pkl'), 'wb') as f:
            pickle.dump(self.params, f) 
        tr_data = self.data_input.get_tr_data_as_single_batch()
        risks_tte = self.model_evaluator.compute_landmarked_cox_risks(
            self.model_tte,
            tr_data, start_time, 'meow'
        )
        
        risks_ttc = self.model_evaluator.compute_landmarked_cox_risks(
            self.model_ttc,
            tr_data, start_time, 'meow'
        )
        
        plt.plot(risks_tte, risks_ttc)
        plt.savefig('risks_correlation_plot.png')
        plt.clf()
        with open(os.path.join(self.params['savedir'], 'risks_tte.pkl'), 'wb') as f:
            pickle.dump(risks_tte, f)

        with open(os.path.join(self.params['savedir'], 'risks_ttc.pkl'), 'wb') as f:
            pickle.dump(risks_ttc, f)
