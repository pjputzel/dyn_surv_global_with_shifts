from main_types.BaseMain import BaseMain
from utils.Diagnostics import Diagnostics
import os
#from models.BasicModelOneTheta import BasicModelOneTheta
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
from models.BasicModelThetaPerStep import BasicModelThetaPerStep
from models.DeltaIJModel import DeltaIJModel
from models.LinearDeltaIJModel import LinearDeltaIJModel
from models.LinearThetaIJModel import LinearThetaIJModel
from models.DummyGlobalModel import DummyGlobalModel
from models.ConstantDeltaModelLinearRegression import ConstantDeltaModelLinearRegression
from models.EmbeddingConstantDeltaModelLinearRegression import EmbeddingConstantDeltaModelLinearRegression
import pickle
#from utils.MetricsTracker import MetricsTracker
from utils.ParameterParser import ParameterParser
from data_handling.DataInput import DataInput
from evaluation.ModelEvaluator import ModelEvaluator
from plotting.DynamicMetricsPlotter import DynamicMetricsPlotter
import torch

class BasicMain(BaseMain):
    

    def load_data(self):
        data_input = DataInput(self.params['data_input_params'])
        data_input.load_data()
        self.data_input = data_input
        return data_input
    
    def preprocess_data(self, data_input):
        print('no data preprocessing in the basic main') 

    def load_model(self):
        model_type = self.params['model_params']['model_type']
        if model_type == 'theta_per_step':
            self.model = BasicModelThetaPerStep(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
            )
        elif model_type == 'linear_theta_per_step':
            self.model = LinearThetaIJModel(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
            )
            
        elif model_type == 'dummy_global_zero_deltas':
            self.model = DummyGlobalModel(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
            )
       
        elif model_type == 'linear_delta_per_step':
            self.model = LinearDeltaIJModel(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
            )
     
        elif model_type == 'linear_constant_delta':
            self.model = ConstantDeltaModelLinearRegression(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
        )
        elif model_type == 'embedding_linear_constant_delta':
            self.model = EmbeddingConstantDeltaModelLinearRegression(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
        )
        elif model_type == 'delta_per_step':
            self.model = DeltaIJModel(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
            )
            
        else:
            raise ValueError('Model type %s not recognized' %(model_type))
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
        
            
        
        
        
    
    def save_results(self, results_tracker):
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])

        with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)

        with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(self.params['savedir'], 'params.pkl'), 'wb') as f:
            pickle.dump(self.params, f) 


