import os
from main_types.BasicMain import BasicMain
import pickle
import torch
import numpy as np

class SaveCovidMedsMain:

    def __init__(self, params):
        self.basic_main = BasicMain(params)
        self.params = params        

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        data_input = self.basic_main.load_data()
        self.basic_main.preprocess_data(data_input)
        meds = data_input.covariate_trajectories[0:10, 0, data_input.num_cont_dynamic_covs:data_input.num_cont_dynamic_covs + 100]
        np.savetxt('example_meds.csv', meds.detach().numpy(), delimiter=',')

#    def load_model(self):
#        print(self.basic_main.params['savedir'])
#        model_path = os.path.join(
#            self.basic_main.params['savedir'],
#            'model.pkl'
#        )
#        with open(model_path, 'rb') as f:
#            model = pickle.load(f)
#        return model
