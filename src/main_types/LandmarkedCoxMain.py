class LandmarkedCoxMain(BasicMain):

    def __init__(self, params):
        self.params = params
        self.params['savedir'] = os.path.join(\
            params['savedir_pre'], 'landmarked_cox'
        )
        if not os.path.exists(self.params['savedir']):
            os.makedirs(self.params['savedir'])

    def load_model(self):
        # model will have a series of cox models at each eval time
        # stored in a dictionary mapping eval time to the matching
        # landmarked model
        self.model = LandmarkedCoxModel(
            self.params['model_params'], 
            self.params['eval_params']['start_times']
        )
        return self.model 

    def train_model(self, model, data_input):
        # just handling the training here
        for landmark_time in self.params['eval_params']['start_times']:
            landmark_data = data_input.get_landmarked_dataset(landmark_time)
            self.train_single_cox_model(
                model.landmarked_cox_models[landmark_time],
                landmark_data
            )
        # check that this is correct todo when not needing diagnositcs
        self.diagnostics = {}
        return diagnostics

    def train_single_cox_model(self, model, data):
        # whatever the call is goes here

    
    def evaluate_model(self, model, data_input, diagnostics):
        self.model_evaluator = ModelEvaluator(
            self.params['eval_params'],
            self.params['train_params']['loss_params'],
            self.params['model_params']['model_type']
        )
        
