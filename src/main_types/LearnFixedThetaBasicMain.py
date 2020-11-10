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
import copy 

# Since learning the params alone will typically have smoother training
# we can take bigger steps in the descent than training the RNN
LR_THETA_RATIO = 1000

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
        config_for_global_param_training = copy.deepcopy(self.params)
        config_for_global_param_training['model_params']['model_type'] = 'dummy_global'
        config_for_global_param_training['train_params']['learning_rate'] = \
            LR_THETA_RATIO  * self.params['train_params']['learning_rate']
        model_trainer = BasicModelTrainer(
            config_for_global_param_training['train_params'],
            config_for_global_param_training['model_params']['model_type']
        )
        _ = model_trainer.train_model(self.dummy_global_model, data_input)

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
    
#    def freeze_hidden_state_and_cov_pred_params(self, model):
#        if self.params['model_params']['model_type'] == 'one_theta':
#            model.freeze_rnn_parameters()
#            model.freeze_cov_pred_parameters()
#        elif self.params['model_params']['model_type'] == 'theta_per_step':
#            model.freeze_rnn1_parameters()
#            model.freeze_cov_pred_parameters()
#        else:
#            raise ValueError('Model type %s not found' %self.params['model_params']['model_type']) 

