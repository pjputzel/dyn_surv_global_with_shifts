from loss.ExponentialLogProbCalculatorThetaPerStep import ExponentialLogProbCalculatorThetaPerStep
from loss.GammaLogProbCalculatorConstantDelta import GammaLogProbCalculatorConstantDelta
from loss.RayleighLogProbCalculatorConstantDelta import RayleighLogProbCalculatorConstantDelta
from loss.RayleighLogProbCalculatorDeltaIJ import RayleighLogProbCalculatorDeltaIJ
from loss.RayleighLogProbCalculatorThetaIJ import RayleighLogProbCalculatorThetaIJ
from loss.RayleighLogProbCalculatorGlobalParam import RayleighLogProbCalculatorGlobalParam
from loss.RegularizationCalculatorThetaPerStep import RegularizationCalculatorThetaPerStep
from loss.WeibullLogProbCalculatorThetaPerStep import WeibullLogProbCalculatorThetaPerStep
from loss.WeibullLogProbCalculatorDeltaIJ import WeibullLogProbCalculatorDeltaIJ
from loss.Chen2000LogProbCalculatorDeltaIJ import Chen2000LogProbCalculatorDeltaIJ
from loss.EMWELogProbCalculatorDeltaIJ import EMWELogProbCalculatorDeltaIJ
from loss.WeibullLogProbCalculatorConstantDelta import WeibullLogProbCalculatorConstantDelta
from loss.RegularizationCalculatorConstantDelta import RegularizationCalculatorConstantDelta
from loss.RegularizationCalculatorDeltaIJ import RegularizationCalculatorDeltaIJ
from loss.GompertzLogProbCalculatorDeltaIJ import GompertzLogProbCalculatorDeltaIJ

class LossCalculator:
    
    def __init__(self, loss_params, model_type):
        self.params = loss_params
        self.model_type = model_type
        self.init_logprob_and_regularization()

    def init_logprob_and_regularization(self):
        dist_type = self.params['distribution_type']
        model_type = self.model_type
        if model_type == 'RNN_delta_per_step' or  model_type == 'dummy_global_zero_deltas' or model_type == 'linear_delta_per_step' or model_type == 'linear_delta_per_step_num_visits_only':
            if dist_type == 'weibull':
                self.logprob_calculator = WeibullLogProbCalculatorDeltaIJ(self.params)
            elif dist_type == 'rayleigh':
                self.logprob_calculator = RayleighLogProbCalculatorDeltaIJ(self.params)
            elif dist_type == 'chen2000':
                self.logprob_calculator = Chen2000LogProbCalculatorDeltaIJ(self.params)
            elif dist_type == 'emwe':
                self.logprob_calculator = EMWELogProbCalculatorDeltaIJ(self.params)

            elif dist_type == 'gompertz':
                self.logprob_calculator = GompertzLogProbCalculatorDeltaIJ(self.params)
            else:
                raise ValueError('Distribution type %s not recognized' %dist_type)
            self.reg_calculator = RegularizationCalculatorDeltaIJ(self.params)

        elif model_type == 'theta_per_step' or model_type == 'linear_theta_per_step':
            if dist_type == 'rayleigh':
                self.logprob_calculator = RayleighLogProbCalculatorThetaIJ(self.params)
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
        
        elif model_type == 'dummy_global':
            # note this is different from dummy_global zero deltas which uses the 
            # 'pred per step' loss. This uses a standard survival log likelihood
            if dist_type == 'rayleigh':
                self.logprob_calculator = RayleighLogProbCalculatorGlobalParam(self.params)
            else:
                raise NotImplementedError('Distribution type %s not yet implemented with dummy global model' %model_type)
            # in this case this is just fed zeros
            self.reg_calculator = RegularizationCalculatorDeltaIJ(self.params)

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

    
        
