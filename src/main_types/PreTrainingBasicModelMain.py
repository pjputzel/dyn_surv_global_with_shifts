from main_types.BaseMain import BaseMain
from utils.Diagnostics import Diagnostics
import os
from models.BasicModelOneTheta import BasicModelOneTheta
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
import pickle
#from utils.MetricsTracker import MetricsTracker
from utils.ParameterParser import ParameterParser
from data_loading.DataInput import DataInput
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
        self.model = BasicModelOneTheta(self.params['model_params'], self.params['diagnostic_params']['distribution_type'])
        return self.model

    def train_model(self, model, data_input):
        model_trainer = BasicModelTrainer(self.params['train_params'])
        diagnostics = Diagnostics(self.params['diagnostic_params'])
        print('PRETRAINING')
        pre_train_diagnostics = model_trainer.train_model(model, data_input, diagnostics, loss_type='reg_only')
        model_trainer.params['learning_rate'] = .01 #* model_trainer.params['learning_rate']
        print('TRAINING LOG-LOSS')
        diagnostics = model_trainer.train_model(model, data_input, diagnostics, loss_type='total_loss')
        diagnostics.unshuffle_results(data_input.unshuffled_idxs)
        return diagnostics
    
    def save_results(self, results_tracker):
        with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)
        with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        


