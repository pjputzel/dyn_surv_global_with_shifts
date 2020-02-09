class BaseMain:
    def __init__(self, params):
        self.params = params

    def main(self):
        data_input = self.load_data()
        self.preprocess_data(data_input)
        model = self.load_model()
        results_tracker = self.train_model(model, data_input)
        self.save_results(results_tracker)

    def load_data(self):
        # TODO: should be the same for almost all runs so just make this function defined here
        pass 
#        raise NotImplementedError('The Main class must be subclassed and have each function defined in the subclass')

    def preprocess_data(self, data_input): 
        raise NotImplementedError('The Main class must be subclassed and have each function defined in the subclass')

    def load_model(self):
        raise NotImplementedError('The Main class must be subclassed and have each function defined in the subclass')
    
    def train_model(self, model, data_input):
        raise NotImplementedError('The Main class must be subclassed and have each function defined in the subclass')

    def save_results(self, results_tracker):
        raise NotImplementedError('The Main class must be subclassed and have each function defined in the subclass')




