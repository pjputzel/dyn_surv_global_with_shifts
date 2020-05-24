from loss.ExponentialLogProbCalculator import ExponentialLogProbCalculator
from loss.RegularizationCalculator import RegularizationCalculator

class LossCalculator:
    
    def __init__(self, loss_params):
        self.params = loss_params
        self.init_logprob_and_regularization()

    def init_logprob_and_regularization(self):
        dist_type = self.params['distribution_type']
        if dist_type == 'exponential':
            self.logprob_calculator = ExponentialLogProbCalculator(self.params)
        elif dist_type == 'weibull':
            self.logprob_calculator = WeibullLogProbCalculator(self.params)
        else:
            raise ValueError('Distribution type %s not recognized' %dist_type)

        # if need more than one type of regularization calculator then 
        # add another switch here
        self.reg_calculator = RegularizationCalculator(self.params)

    def compute_batch_loss(self,
        global_theta,
        pred_params, hidden_states, 
        step_ahead_cov_preds, batch
    ):

        logprob = self.logprob_calculator(pred_params, batch)
        print(logprob)
        reg = self.reg_calculator(
            global_theta,
            pred_params, hidden_states, 
            step_ahead_cov_preds, batch, ret_each_term=False
        )
        total_loss = logprob + reg
        return total_loss, logprob, reg

    
        
