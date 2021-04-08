from main_types.BasicMain import BasicMain 
from utils.Diagnostics import Diagnostics
import os
import pickle
from evaluation.ModelEvaluator import ModelEvaluator
import torch
import numpy as np
from models.LandmarkedCoxModel import LandmarkedCoxModel

class LandmarkedCoxMain(BasicMain):

    def __init__(self, params):
        self.params = params
        self.params['savedir'] = os.path.join(\
            params['savedir_pre'], 'landmarked_cox'
        )
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])
        self.device = torch.device(self.params['device'])

    def load_model(self):
        # model will have a series of cox models at each eval time
        # stored in a dictionary mapping eval time to the matching
        # landmarked model
        self.model = LandmarkedCoxModel(
            self.params['model_params'], 
            self.params['eval_params']['dynamic_metrics']['start_times']
        )
        return self.model 

    def train_model(self, model, data_input):
        # just handling the training here
        for landmark_time in self.params['eval_params']['dynamic_metrics']['start_times']:
            landmark_data = data_input.get_tr_landmarked_dataset(landmark_time)
            self.train_single_cox_model(
                model.models[landmark_time],
                landmark_data
            )
        # check that this is correct todo when not needing diagnositcs
        self.diagnostics = Diagnostics(self.params['train_params']['diagnostic_params'])
        return self.diagnostics

    def train_single_cox_model(self, model, data):
        event_indicators = (~data['censoring'].astype(bool)).astype(int)
        model.fit(
            data['covs'], data['event_times'], event_indicators,
            max_iter=self.params['train_params']['max_iter'],
            lr=self.params['train_params']['learning_rate'],
            tol=self.params['train_params']['conv_thresh'],
            l2_reg=self.params['train_params']['loss_params']['cox_l2_reg'],
            verbose=False
        )

    
    def evaluate_model(self, model, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            self.params['model_params']['model_type']
        )
        self.model_evaluator.evaluate_model(
            model, data_input, diagnostics
        )
