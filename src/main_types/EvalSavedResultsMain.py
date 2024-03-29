import os
from main_types.BasicMain import BasicMain
import pickle
import torch
import numpy as np
from utils.Diagnostics import Diagnostics

class EvalSavedResultsMain:

    def __init__(self, params):
        self.basic_main = BasicMain(params)
        self.params = params        

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        data_input = self.basic_main.load_data()
        self.basic_main.preprocess_data(data_input)
        model = self.basic_main.load_model()
        try:
            # only will work for pytorch defined models
            model.eval()
        except:
            pass
        diagnostics = self.basic_main.evaluate_model(model, data_input, Diagnostics({}))
        with open(os.path.join(self.basic_main.params['savedir'], 'tracker_updated.pkl'), 'wb') as f:
           pickle.dump(diagnostics, f) 

#    def load_model(self):
#        print(self.basic_main.params['savedir'])
#        model_path = os.path.join(
#            self.basic_main.params['savedir'],
#            'model.pkl'
#        )
#        with open(model_path, 'rb') as f:
#            model = pickle.load(f)
#        return model
