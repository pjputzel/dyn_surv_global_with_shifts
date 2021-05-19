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
LR_THETA_RATIO = 1. # 100 # for rayleigh

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
        diagnostics = super().train_model(model, data_input)
#        model_trainer = BasicModelTrainer(
#            self.params['train_params'],
#            self.params['model_params']['model_type']
#        )
#        diagnostics = model_trainer.train_model(model, data_input)
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
            device = next(self.model.parameters()).device
            if next(self.model.parameters()).is_cuda:
                # note that self.model isn't being trained here
                # just using it since it and datainput are put onto the
                # appropriate device at the same time
                self.dummy_global_model.to('cpu')
                data_input.to_device('cpu')
            # note: should probably just have a config file for global param
            # training saved and just load it here based on the global param type
            config_for_global_param_training = copy.deepcopy(self.params)
            config_for_global_param_training['model_params']['model_type'] = 'dummy_global'
            config_for_global_param_training['train_params']['learning_rate'] = \
                LR_THETA_RATIO  * self.params['train_params']['learning_rate']
            config_for_global_param_training['train_params']['batch_size'] = 100000
            config_for_global_param_training['train_params']['max_iter'] = 10000
            config_for_global_param_training['train_params']['loss_params']['l1_reg'] = 0
            config_for_global_param_training['train_params']['loss_params']['l2_reg'] = 0
 #           print('If any regularization is added besides l1, then global model training needs to be updated to turn off that regularization!')
            model_trainer = BasicModelTrainer(
                config_for_global_param_training['train_params'],
                config_for_global_param_training['model_params']['model_type']
            )
            _ = model_trainer.train_model(self.dummy_global_model, data_input)
            if not device == 'cpu':
                data_input.to_device(device) 
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

