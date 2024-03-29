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
        split = self.params['data_input_params']['data_loading_params']['paths'].split('/')
        data_dir = ''
        for i in range(len(split) - 1):
            data_dir += split[i] + '/'
        self.data_dir = data_dir        
        if self.params['data_input_params']['dataset_name'] == 'pbc2':
            # as mentioned in our paper we tuned a scale hyperparameter for this
            # dataset so the evaluation times also need to be rescaled
            timescale = self.params['data_input_params']['data_loading_params']['timescale']
            self.params['eval_params']['dynamic_metrics']['start_times'] = [\
                time/timescale
                for time in self.params['eval_params']['dynamic_metrics']['start_times']
            ]
            
            self.params['eval_params']['dynamic_metrics']['time_step'] = self.params['eval_params']['dynamic_metrics']['time_step']/timescale
            self.params['eval_params']['dynamic_metrics']['window_length'] = self.params['eval_params']['dynamic_metrics']['window_length']/timescale
            self.params['savedir'] = self.params['savedir'] + '_timescale%d' %timescale

        data_path = os.path.join(data_dir, 'data_input.pkl')
        if 'data_input.pkl' in os.listdir(data_dir):
            print('Loading saved data input object')
            with open(data_path, 'rb') as f:
                data_input = pickle.load(f) 
        else:
            data_input = DataInput(self.params['data_input_params'])
            data_input.load_data()
            self.data_input = data_input
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
        print(results_tracker.eval_metrics['brier_score']['te']['values'])
        print(results_tracker.eval_metrics['c_index']['te']['values'])
        print(results_tracker.eval_metrics['c_index_truncated_at_S']['te']['values'])
        
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
        print('evaluating!')

class EvaluateNumEventsRankingMain(ModelFreeRankingMain):

    def evaluate_model(self, model_type, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            model_type
        )
        # evaluate with num_events_ranking
        self.model_evaluator.evaluate_model('num_events_ranking', data_input, diagnostics)

class EvaluateFraminghamRankingMain(ModelFreeRankingMain):

    def evaluate_model(self, model_type, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            model_type, verbose=True
        )
        # evaluate with num_events_ranking
        self.model_evaluator.evaluate_model('framingham', data_input, diagnostics)

class EvaluateBrierScoreBaseRatesMain(ModelFreeRankingMain):

    def evaluate_model(self, model_type, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            model_type
        )
        self.model_evaluator.evaluate_model('brier_base_rate', data_input, diagnostics)
    
