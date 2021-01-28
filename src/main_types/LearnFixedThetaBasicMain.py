from main_types.BaseMain import BaseMain
from main_types.BasicMain import BasicMain
from utils.Diagnostics import Diagnostics
import os
from models.BasicModelThetaPerStep import BasicModelThetaPerStep
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
from models.DummyGlobalModel import DummyGlobalModel
import pickle
#from utils.MetricsTracker import MetricsTracker
from utils.ParameterParser import ParameterParser
from data_handling.DataInput import DataInput
import torch
import torch.nn as nn
import copy 

# Since learning the params alone will typically have smoother training
# we can take bigger steps in the descent than training the RNN
# global model generally converges, and this doesn't seem to affect the optimum
# just the speed of the global training.
LR_THETA_RATIO = 100

class LearnFixedThetaBasicMain(BasicMain):
    
    def __init__(self, params):
        super().__init__(params)
        torch.set_default_dtype(torch.float64)

    def load_model(self):
        self.model = super().load_model()
        self.dummy_global_model = DummyGlobalModel(
            self.params['model_params'],
            self.params['train_params']['loss_params']['distribution_type']
        )
        return self.model

    def train_model(self, model, data_input):
        print('TRAINING GLOBAL PARAM ONLY')
        self.train_global_model(data_input)
        print('FREEZING GLOBAL PARAM, TRAINING SHIFTS')
        self.model.set_and_freeze_global_param(
            self.dummy_global_model.get_global_param()
        )
        model_trainer = BasicModelTrainer(
            self.params['train_params'],
            self.params['model_params']['model_type']
        )
        diagnostics = model_trainer.train_model(model, data_input)
        return diagnostics
    
    def train_global_model(self, data_input):
        theta_filename = 'global_theta_%s.pkl' %self.params['train_params']['loss_params']['distribution_type']
        saved_theta_path = os.path.join(self.data_dir, theta_filename)
        if os.path.exists(saved_theta_path):
            with open(saved_theta_path, 'rb') as f:
                saved_global_theta = pickle.load(f)
            self.dummy_global_model.set_global_param(saved_global_theta)
            print('Loading pre-saved global theta with value:', saved_global_theta)
        else:
            config_for_global_param_training = copy.deepcopy(self.params)
            config_for_global_param_training['model_params']['model_type'] = 'dummy_global'
            config_for_global_param_training['train_params']['learning_rate'] = \
                LR_THETA_RATIO  * self.params['train_params']['learning_rate']
            model_trainer = BasicModelTrainer(
                config_for_global_param_training['train_params'],
                config_for_global_param_training['model_params']['model_type']
            )
            _ = model_trainer.train_model(self.dummy_global_model, data_input)
            with open(saved_theta_path, 'wb') as f:
                pickle.dump(self.dummy_global_model.get_global_param(), f)

#    def freeze_hidden_state_and_cov_pred_params(self, model):
#        if self.params['model_params']['model_type'] == 'one_theta':
#            model.freeze_rnn_parameters()
#            model.freeze_cov_pred_parameters()
#        elif self.params['model_params']['model_type'] == 'theta_per_step':
#            model.freeze_rnn1_parameters()
#            model.freeze_cov_pred_parameters()
#        else:
#            raise ValueError('Model type %s not found' %self.params['model_params']['model_type']) 

