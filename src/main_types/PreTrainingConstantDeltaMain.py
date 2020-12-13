from main_types.BaseMain import BaseMain
from utils.Diagnostics import Diagnostics
import os
#from models.BasicModelOneTheta import BasicModelOneTheta
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
from models.BasicModelThetaPerStep import BasicModelThetaPerStep
from models.ConstantDeltaModelLinearRegression import ConstantDeltaModelLinearRegression
from evaluation.ModelEvaluator import ModelEvaluator
import pickle
#from utils.MetricsTracker import MetricsTracker
from utils.ParameterParser import ParameterParser
from data_handling.DataInput import DataInput
import torch

class PreTrainingConstantDeltaMain(BaseMain):
    
    def __init__(self, params):
        super().__init__(params)
        torch.set_default_dtype(torch.float64)

    def load_data(self):
        data_input = DataInput(self.params['data_input_params'])
        data_input.load_data()
        self.data_input = data_input
        return data_input
    
    def preprocess_data(self, data_input):
        print('no data preprocessing in the basic main') 

    def load_model(self):
        if self.params['model_params']['model_type'] == 'theta_per_step':
            self.model = BasicModelThetaPerStep(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
            )
        elif self.params['model_params']['model_type'] == 'linear_constant_delta':
            self.model = ConstantDeltaModelLinearRegression(
                self.params['model_params'],
                self.params['train_params']['loss_params']['distribution_type']
        )
        else:
            raise ValueError('Model type %s not recognized' %(self.params['model_params']['model_type']))
        return self.model

    def train_model(self, model, data_input):
        print('Pretraining with fixed zero deltas')
        model_trainer = BasicModelTrainer(
            self.params['train_params'],
            self.params['model_params']['model_type']
        )
        #if self.params['train_params']['loss_params']['distribution_type'] == 'rayleigh':
        #    event_times = data_input.event_times
        #    print(torch.sum(event_times**2))
        #    scale = ( 1./(event_times.shape[0] * 2) ) * torch.sum(event_times**2)
        #    model.global_param_logspace = torch.nn.Parameter(torch.log(-scale).unsqueeze(0))
        #    data_input.make_randomized_batches(self.params['train_params']['batch_size'])
        model.fix_deltas_to_zero()
        model_trainer.params['learning_rate'] = self.params['train_params']['pretraining']['pretraining_lr']
        model_trainer.params['max_iter'] = self.params['train_params']['pretraining']['pretraining_max_iter']
        pre_train_diagnostics = model_trainer.train_model(model, data_input)
        print(model.get_global_param())
        print(model(data_input.tr_batches[0])[0])

        print('Training deltas with fixed global params')
        model.unfix_deltas_to_zero()
        model.freeze_global_param()
        model_trainer.params['learning_rate'] = self.params['train_params']['pretraining']['regular_training_lr']
        model_trainer.params['max_iter'] = self.params['train_params']['pretraining']['regular_training_max_iter']
        diagnostics = model_trainer.train_model(model, data_input)
        print(model.get_global_param())
        deltas = model(data_input.tr_batches[0])[0]
        print('Deltas:', deltas)
        print('Shifted cov times:', torch.max(data_input.tr_batches[0].cov_times, dim=1)[0] + deltas.squeeze(1))
        #diagnostics.unshuffle_results(data_input.unshuffled_idxs)
        return diagnostics
    
    def evaluate_model(self, model, data_input, diagnostics):
        model.eval()
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            self.params['model_params']['model_type']
        )
        self.model_evaluator.evaluate_model(model, data_input, diagnostics)
    
    def save_results(self, results_tracker):
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])

        with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)

        with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(os.path.join(self.params['savedir'], 'params.pkl'), 'wb') as f:
            pickle.dump(self.params, f) 


        
        


