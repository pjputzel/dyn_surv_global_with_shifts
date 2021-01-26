import sys
sys.path.append('../data/')
import numpy as np
import os
import pickle
import torch
from data_handling.DataInput import DataInput
from loss.LossCalculator import LossCalculator
from utils.Diagnostics import Diagnostics
from evaluation.ModelEvaluator import ModelEvaluator
from utils.ParameterParser import ParameterParser
from plotting.DynamicMetricsPlotter import DynamicMetricsPlotter
from main_types.BaseMain import BaseMain

class ModelFreeRankingMain(BaseMain):

    def load_data(self):
        data_input = DataInput(self.params['data_input_params'])
        data_input.load_data()
        self.data_input = data_input
        print(data_input.covariate_trajectories)
        return data_input
    
    def preprocess_data(self, data_input):
        print('no data preprocessing in the basic main') 
    
    def load_model(self):
        # just pass a string here to fit
        # the function definition
        # no model being trained
        model_type = 'dummy_global_zero_deltas'
        return model_type

    
    def train_model(self, model, data_input):
        diagnostics = Diagnostics(
            self.params['train_params']['diagnostic_params']
        )
        return diagnostics

    def evaluate_model(self, model_type, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            model_type
        )
        # evaluate with no model
        self.model_evaluator.evaluate_model('cov_times_ranking', data_input, diagnostics)
        
    def plot_results(self, model_type, data_input, diagnostics):
        metrics_evaluated = self.params['eval_params']['eval_metrics']
        plotter = DynamicMetricsPlotter(
            self.params['plot_params'], self.params['savedir']
        )
        plotter.make_and_save_dynamic_eval_metrics_plots(diagnostics.eval_metrics)
        
    def save_results(self, results_tracker):
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])

        with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)
        
        with open(os.path.join(self.params['savedir'], 'params.pkl'), 'wb') as f:
            pickle.dump(self.params, f) 

class EvaluateCovTimesRankingMain(ModelFreeRankingMain):

    def evaluate_model(self, model_type, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            model_type
        )
        # evaluate with cov_times_ranking
        self.model_evaluator.evaluate_model('cov_times_ranking', data_input, diagnostics)

class EvaluateNumEventsRankingMain(ModelFreeRankingMain):

    def evaluate_model(self, model_type, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            model_type
        )
        # evaluate with num_events_ranking
        self.model_evaluator.evaluate_model('num_events_ranking', data_input, diagnostics)

