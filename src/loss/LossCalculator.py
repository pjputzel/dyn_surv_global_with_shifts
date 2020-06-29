from loss.ExponentialLogProbCalculatorThetaPerStep import ExponentialLogProbCalculatorThetaPerStep
from loss.GammaLogProbCalculatorConstantDelta import GammaLogProbCalculatorConstantDelta
from loss.RayleighLogProbCalculatorConstantDelta import RayleighLogProbCalculatorConstantDelta
from loss.RayleighLogProbCalculatorDeltaIJ import RayleighLogProbCalculatorDeltaIJ
from loss.RegularizationCalculatorThetaPerStep import RegularizationCalculatorThetaPerStep
from loss.WeibullLogProbCalculatorThetaPerStep import WeibullLogProbCalculatorThetaPerStep
from loss.WeibullLogProbCalculatorConstantDelta import WeibullLogProbCalculatorConstantDelta
from loss.RegularizationCalculatorConstantDelta import RegularizationCalculatorConstantDelta


class LossCalculator:
    
    def __init__(self, loss_params, model_type):
        self.params = loss_params
        self.model_type = model_type
        self.init_logprob_and_regularization()

    def init_logprob_and_regularization(self):
        dist_type = self.params['distribution_type']
        model_type = self.model_type
        if model_type == 'delta_per_step':
            if dist_type == 'exponential':
                self.logprob_calculator = ExponentialLogProbCalculatorThetaPerStep(self.params)
            elif dist_type == 'weibull':
                self.logprob_calculator = WeibullLogProbCalculatorThetaPerStep(self.params)
            elif dist_type == 'rayleigh':
                self.logprob_calculator = RayleighLogProbCalculatorDeltaIJ(self.params)
            else:
                raise ValueError('Distribution type %s not recognized' %dist_type)
            self.reg_calculator = RegularizationCalculatorThetaPerStep(self.params)
        elif model_type == 'linear_constant_delta' or model_type == 'embedding_linear_constant_delta':
            if dist_type == 'weibull':
                self.logprob_calculator = WeibullLogProbCalculatorConstantDelta(self.params)
            elif dist_type == 'rayleigh':
                self.logprob_calculator = RayleighLogProbCalculatorConstantDelta(self.params)
            elif dist_type == 'gamma':
                self.logprob_calculator = GammaLogProbCalculatorConstantDelta(self.params)
            else:
                raise ValueError('Distribution type %s not recognized' %dist_type)
            self.reg_calculator = RegularizationCalculatorConstantDelta(self.params)
        else:
            raise ValueError('Model type %s not recognized' %model_type)            


    def compute_batch_loss(self,
        global_theta,
        pred_params, hidden_states, 
        step_ahead_cov_preds, batch
    ):

        logprob = self.logprob_calculator(pred_params, batch, global_theta=global_theta)
        reg = self.reg_calculator(
            global_theta,
            pred_params, hidden_states, 
            step_ahead_cov_preds, batch, ret_each_term=False
        )
        if reg < 0:
            print(reg)
        total_loss = -logprob + reg
        return total_loss, reg, logprob

    
        
