import sys
sys.path.append('../data/')
from main_types.BaseMain import BaseMain
from utils.Diagnostics import Diagnostics
import os
#from models.BasicModelOneTheta import BasicModelOneTheta
from models import models_dict
#from models.GlobalPlusEpsModel import GlobalPlusEpsModel
#from models.BasicModelTrainer import BasicModelTrainer
#from models.BasicModelThetaPerStep import BasicModelThetaPerStep
#from models.DeltaIJModel import DeltaIJModel
#from models.LinearDeltaIJModel import LinearDeltaIJModel
#from models.LinearThetaIJModel import LinearThetaIJModel
#from models.LinearDeltaIJModelNumVisitsOnly import LinearDeltaIJModelNumVisitsOnly
#from models.DummyGlobalModel import DummyGlobalModel
#from models.ConstantDeltaModelLinearRegression import ConstantDeltaModelLinearRegression
#from models.EmbeddingConstantDeltaModelLinearRegression import EmbeddingConstantDeltaModelLinearRegression

import pickle
#from utils.MetricsTracker import MetricsTracker
from utils.ParameterParser import ParameterParser
from data_handling.DataInput import DataInput
from evaluation.ModelEvaluator import ModelEvaluator
from plotting.DynamicMetricsPlotter import DynamicMetricsPlotter
from plotting.ResultsPlotterSynth import ResultsPlotterSynth
import torch

class BasicMain(BaseMain):
    
    def __init__(self, params):
        self.params = params

        if params['model_params']['model_type'] == 'RNN_delta_per_step':
            self.params['savedir'] = os.path.join(\
                params['savedir_pre'], 
                '%s/%s_hdim%d' %(
                        params['train_params']['loss_params']['distribution_type'],
                        params['model_params']['model_type'],
                        params['model_params']['hidden_dim']
                )
            )
        else:
            self.params['savedir'] = os.path.join(\
                params['savedir_pre'], 
                '%s/%s' %(
                        params['train_params']['loss_params']['distribution_type'],
                        params['model_params']['model_type'],
                )
            )
            
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])
        
        self.device = torch.device(self.params['device'])
        

    def load_data(self):
        split = self.params['data_input_params']['data_loading_params']['paths'].split('/')
        data_dir = ''
        for i in range(len(split) - 1):
            data_dir += split[i] + '/'
        self.data_dir = data_dir        

        data_path = os.path.join(data_dir, 'data_input.pkl')
        if 'data_input.pkl' in os.listdir(data_dir):
            print('Loading saved data input object')
            with open(data_path, 'rb') as f:
                data_input = pickle.load(f) 
        else:
            data_input = DataInput(self.params['data_input_params'])
            data_input.load_data()
            try:
                with open(data_path, 'wb') as f:
                    pickle.dump(data_input, f)
            except:
                os.remove(data_path)
                print('processed data too large to pickle')
        self.data_input = data_input
        data_input.to_device(self.device)
        return data_input
    
    def preprocess_data(self, data_input):
        print('no data preprocessing in the basic main') 

    def load_model(self):
        model_type = self.params['model_params']['model_type']
        try:
            self.model = models_dict[model_type](
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
            )
        except:
            raise ValueError('Model type %s not recognized' %(model_type))
        self.model.to(self.device)
        return self.model

    def train_model(self, model, data_input):
        model_trainer = BasicModelTrainer(
            self.params['train_params'],
            self.params['model_params']['model_type']
        )
        diagnostics = model_trainer.train_model(model, data_input)
        #diagnostics.unshuffle_results(data_input.unshuffled_idxs)
        return diagnostics

    def evaluate_model(self, model, data_input, diagnostics):
        data_input.to_device('cpu')
        model.to('cpu')
        model.eval()
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            self.params['model_params']['model_type']
        )
        self.model_evaluator.evaluate_model(model, data_input, diagnostics)
        
    def plot_results(self, model, data_input, diagnostics):
        metrics_evaluated = self.params['eval_params']['eval_metrics']
        plotter = DynamicMetricsPlotter(
            self.params['plot_params'], self.params['savedir']
        )
        plotter.make_and_save_dynamic_eval_metrics_plots(diagnostics.eval_metrics)
        if self.params['data_input_params']['dataset_name'] == 'simple_synth':
            plotter = ResultsPlotterSynth(model, self.params['plot_params'])
            plotter.plot_event_time_samples_from_learned_model(
                self.params['data_input_params']['data_loading_params']['paths'],
                self.params['savedir']
            )
    
    def save_results(self, results_tracker):

        try:
            with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
                pickle.dump(results_tracker, f)
        except:
            print('Tracker save file too large to save!')

        with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(self.params['savedir'], 'params.pkl'), 'wb') as f:
            pickle.dump(self.params, f) 


