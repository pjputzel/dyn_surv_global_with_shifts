from pysurvival.models.survival_forest import RandomSurvivalForestModel

class LandmarkedRFModel:
    
    def __init__(self, model_params, start_times):
        self.params = model_params
        num_trees = model_params['RF_num_trees']
        self.models = {
            time: RandomSurvivalForestModel(num_trees=num_trees) 
            for time in start_times
        }
