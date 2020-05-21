

class LossCalculator:
    
    def __init__(self, loss_params):
        self.params = loss_params
        self.init_logprob_and_regularization()

    def init_logprob_and_regularization(self):
        logprob_params = self.params['logprob_params']
        dist_type = self.params['dist_type']
        if dist_type == 'exponential':
            self.logprob_calculator = ExponentialLogProbCalculator(logprob_params)
        elif dist_type == 'weibull':
            self.logprob_calculator = WeibullLogProbCalculator(logprob_params)
        else:
            raise ValueError('Distribution type %s not recognized' %dist_type)

        reg_params = self.params['reg_params']
        # if need more than one type of regularization calculator then 
        # add another switch here
        self.reg_calculator = RegularizationCalculator(reg_params)

    def compute_batch_loss(self, pred_params, hidden_states, batch):
        logprob = self.logprob_calculator(model, data_input) 
        reg = self.reg_calculator(model, data_input, ret_each_term=False)
        total_loss = logprob + reg
        return total_loss, logprob, reg

    
        
