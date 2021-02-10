from pysurvival.models.semi_parametric import CoxPHModel

class LandmarkedCoxModel:

    def __init__(self, model_params, start_times):
        self.params = model_params
        self.models = {time:CoxPHModel() for time in start_times}
