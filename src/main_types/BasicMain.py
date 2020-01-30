import BaseMain
import BasicModel
import BasicModelTrainer
import pickle
import utils.MetricsTracker
import utils.ParameterParser

class BasicMain(BaseMain):
    
    def __init__(self, params):
        super(self, BasicMain).__init__(params)

    def load_data(self):
        data_input = DataInput(params['data_input_params'])
        return data_input
    
    def preprocess_data(self, data_input):
        print('no data preprocessing in the basic main') 

    def load_model(self):
        model = BasicModel(self.params['model_params'])
        return model

    def train_model(self, model, data_input):
        model_trainer = BasicModelTrainer(self.params['train_params'])
        tracker = MetricsTracker(model, data_input, self.params['metrics_to_track'])
        model_trainer.train_model(model, data_input, tracker)
        
    
    def save_results(self, results_tracker):
        with open(os.path.join(params['savedir'], 'tracker.pkl'), 'wb') as f:
            pickle.dump(results_tracker, f)


