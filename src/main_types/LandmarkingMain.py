from main_types.BasicMain import BasicMain 
from utils.Diagnostics import Diagnostics
import os
import pickle
from evaluation.ModelEvaluator import ModelEvaluator
import torch
import numpy as np
from models.LandmarkedCoxModel import LandmarkedCoxModel
from models.LandmarkedRFModel import LandmarkedRFModel

class LandmarkingMain(BasicMain):

    def __init__(self, params):
        self.params = params
        self.params['savedir'] = os.path.join(\
            params['savedir_pre'], 
            'landmarked_' + params['model_params']['model_type']
        )
        if self.params['data_input_params']['dataset_name'] == 'pbc2':
            # as mentioned in our paper we tuned a scale hyperparameter for this
            # dataset so the evaluation times also need to be rescaled
            timescale = self.params['data_input_params']['data_loading_params']['timescale']
            self.params['eval_params']['dynamic_metrics']['start_times'] = [\
                time/timescale
                for time in self.params['eval_params']['dynamic_metrics']['start_times']
            ]
            
            self.params['eval_params']['dynamic_metrics']['time_step'] = self.params['eval_params']['dynamic_metrics']['time_step']/timescale
            self.params['eval_params']['dynamic_metrics']['window_length'] = self.params['eval_params']['dynamic_metrics']['window_length']/timescale
            self.params['savedir'] = self.params['savedir'] + '_timescale%d' %timescale
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])
        self.device = torch.device(self.params['device'])

    def load_model(self):
        # model will have a series of cox models at each eval time
        # stored in a dictionary mapping eval time to the matching
        # landmarked model
        if self.params['model_params']['model_type'] == 'landmarked_cox':
            self.model = LandmarkedCoxModel(
                self.params['model_params'], 
                self.params['eval_params']['dynamic_metrics']['start_times']
            )
        elif self.params['model_params']['model_type'] == 'landmarked_RF':
            self.model = LandmarkedRFModel(
                self.params['model_params'],
                self.params['eval_params']['dynamic_metrics']['start_times']
            )
        else:
            raise ValueError('Model type %s not recognized' %self.params['model_params']['model_type'])
        return self.model 

    def train_model(self, model, data_input):
        # just handling the training here
        for landmark_time in self.params['eval_params']['dynamic_metrics']['start_times']:
            landmark_data = data_input.get_tr_landmarked_dataset(landmark_time)
            self.train_single_model(
                model.models[landmark_time],
                landmark_data
            )
        # check that this is correct todo when not needing diagnositcs
        self.diagnostics = Diagnostics(self.params['train_params']['diagnostic_params'])
        return self.diagnostics

    def train_single_model(self, model, data):
        event_indicators = (~data['censoring'].astype(bool)).astype(int)
        model_type = self.params['model_params']['model_type'] 
        if model_type == 'landmarked_cox':
            model.fit(
                data['covs'], data['event_times'], event_indicators,
                max_iter=self.params['train_params']['max_iter'],
                lr=self.params['train_params']['learning_rate'],
                tol=self.params['train_params']['conv_thresh'],
                l2_reg=self.params['train_params']['loss_params']['cox_l2_reg'],
                verbose=False
            )
        else:
            model.fit(
                data['covs'], data['event_times'], event_indicators,
            )
            
    
    def evaluate_model(self, model, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            self.params['model_params']['model_type'],
            verbose=True
        )
        self.model_evaluator.evaluate_model(
            model, data_input, diagnostics 
        )

    def save_results(self, results_tracker):

        try:
            with open(os.path.join(self.params['savedir'], 'tracker.pkl'), 'wb') as f:
                pickle.dump(results_tracker, f)
        except:
            print('Tracker save file too large to save!')
        model_type = self.params['model_params']['model_type']
        if not model_type == 'landmarked_RF':
            # can't pickle survival RF object from pysurv
            with open(os.path.join(self.params['savedir'], 'model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
        
        with open(os.path.join(self.params['savedir'], 'params.pkl'), 'wb') as f:
            pickle.dump(self.params, f) 


