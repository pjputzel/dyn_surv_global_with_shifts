from main_types.BaseMain import BaseMain
from utils.Diagnostics import Diagnostics
import os
#from models.BasicModelOneTheta import BasicModelOneTheta
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
from models.BasicModelThetaPerStep import BasicModelThetaPerStep
import pickle
#from utils.MetricsTracker import MetricsTracker
from utils.ParameterParser import ParameterParser
from data_handling.DataInput import DataInput
import torch

class BasicMain(BaseMain):
    
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
        self.model = BasicModelThetaPerStep(
            self.params['model_params'],
            self.params['train_params']['loss_params']['distribution_type']
        )
        return self.model

    def train_model(self, model, data_input):
        model_trainer = BasicModelTrainer(self.params['train_params'])
        diagnostics = model_trainer.train_model(model, data_input)
        diagnostics.unshuffle_results(data_input.unshuffled_idxs)
        return diagnostics
    
    def save_results(self, results_tracker):
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])
        with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)
        with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        


