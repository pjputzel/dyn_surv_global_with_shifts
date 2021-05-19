import numpy as np
import torch


class BaseMain:

    def __init__(self, params):
        self.params = params

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        data_input = self.load_data()
        self.preprocess_data(data_input)
        model = self.load_model()
        results_tracker = self.train_model(model, data_input)
        # evaluate_model should update the results tracker object to
        # include evaluation metrics
        self.evaluate_model(model, data_input, results_tracker)
        self.save_results(results_tracker)
#        self.plot_results(model, data_input, results_tracker)

    def load_data(self):
        # TODO: should be the same for almost all runs so just make this function defined here
        pass 
#        raise NotImplementedError('The Main class must be subclassed and have each function defined in the subclass')

    def preprocess_data(self, data_input): 
        raise NotImplementedError('The BaseMain class must be subclassed and have each function defined in the subclass')

    def load_model(self):
        raise NotImplementedError('The BaseMain class must be subclassed and have each function defined in the subclass')
    
    def train_model(self, model, data_input):
        raise NotImplementedError('The BaseMain class must be subclassed and have each function defined in the subclass')

    def save_results(self, results_tracker):
        raise NotImplementedError('The BaseMain class must be subclassed and have each function defined in the subclass')

    def evaluate_model(self, model, data_input, results_tracker):
        raise NotImplementedError('The BaseMain class must be subclassed and have each function defined in the subclass')
    
    def plot_results(self, model, data_input, results_tracker):
        raise NotImplementedError('The BaseMain class must be subclassed and have each function defined in the subclass')
         
