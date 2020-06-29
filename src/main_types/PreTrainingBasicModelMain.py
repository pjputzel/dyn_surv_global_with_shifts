from main_types.BaseMain import BaseMain
from utils.Diagnostics import Diagnostics
import os
from models.BasicModelThetaPerStep import BasicModelThetaPerStep
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
import pickle
#from utils.MetricsTracker import MetricsTracker
from utils.ParameterParser import ParameterParser
from data_handling.DataInput import DataInput
import torch

class PreTrainingBasicModelMain(BaseMain):
    
    def __init__(self, params):
        super().__init__(params)
        torch.set_default_dtype(torch.float64)

    def load_data(self):
        data_input = DataInput(self.params['data_input_params'])
        data_input.load_data()
        return data_input
    
    def preprocess_data(self, data_input):
        print('no data preprocessing in the basic main') 

    def load_model(self):
        if self.params['model_params']['model_type'] == 'theta_per_step':
            self.model = BasicModelThetaPerStep(self.params['model_params'], self.params['diagnostic_params']['distribution_type'])
        else:
            raise ValueError('Model type %s not found' %self.params['model_params']['model_type']) 
        return self.model

    def train_model(self, model, data_input):
        model_trainer = BasicModelTrainer(self.params['train_params'])
        if self.params['model_params']['model_type'] == 'one_theta':
            diagnostics = Diagnostics(self.params['diagnostic_params'], one_theta=True)
        else:
            diagnostics = Diagnostics(self.params['diagnostic_params'], one_theta=False)
        print('PRETRAINING')
        pretrain_max_iter = self.params['train_params']['pretraining']['pretraining_max_iter']
        pre_train_diagnostics = model_trainer.train_model(model, data_input, diagnostics, loss_type='reg_only', max_iter=pretrain_max_iter)
        model_trainer.params['learning_rate'] = self.params['train_params']['pretraining']['regular_training_lr'] #* model_trainer.params['learning_rate']
        print('TRAINING LOG-LOSS ONLY, FREEZING HIDDEN STATES')
        self.freeze_hidden_state_and_cov_pred_params(model)
        regular_train_max_iter = self.params['train_params']['pretraining']['regular_training_max_iter']
        diagnostics = model_trainer.train_model(model, data_input, diagnostics, loss_type='log_loss_only', max_iter=regular_train_max_iter)
        diagnostics.unshuffle_results(data_input.unshuffled_idxs)
        return diagnostics
    
    def freeze_hidden_state_and_cov_pred_params(self, model):
        if self.params['model_params']['model_type'] == 'one_theta':
            model.freeze_rnn_parameters()
            model.freeze_cov_pred_parameters()
        elif self.params['model_params']['model_type'] == 'theta_per_step':
            model.freeze_rnn1_parameters()
            model.freeze_cov_pred_parameters()
        else:
            raise ValueError('Model type %s not found' %self.params['model_params']['model_type']) 

    def save_results(self, results_tracker):
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])
        with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)
        with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        


