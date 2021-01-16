import numpy as np
import torch
from loss.loss_calculators import ExponentialLossCalculator
import sys
#from utils.loss_calculators import GGDLossCalculator
#from utils.loss_calculators import RegularizationCalculator
#from utils.loss_calculators import GammaLossCalculator
'''
Computes and holds model diagnostics during training
and eval
'''

class Diagnostics:

    def __init__(self, diagnostic_params):
        self.params = diagnostic_params
        self.epochs = []
        self.pred_params_per_step = []
        self.hidden_states_per_step = []
        self.total_loss_per_step = []
        self.reg_per_step = []
        self.nll_per_step = []

        self.eval_metrics = {}

    def update(self,
        pred_params, hidden_states, total_loss,
        reg, logprob, epoch
    ):
        self.pred_params_per_step.append(pred_params.cpu().detach())
#        if not ignore_hidden_states:
#            self.hidden_states_per_step.append(hidden_states)
#        else:
        self.hidden_states_per_step.append(hidden_states.cpu().detach())

        #print([sys.getsizeof(self.hidden_states_per_step[i].storage()) for i in range(len(self.hidden_states_per_step))])
        #print([sys.getsizeof(self.pred_params_per_step[i].storage()) for i in range(len(self.pred_params_per_step))])

        self.total_loss_per_step.append(total_loss.cpu().detach())
        self.reg_per_step.append(0 if type(reg) is float else reg.cpu().detach())
        self.nll_per_step.append(-logprob.cpu().detach())
        self.epochs.append(epoch)

    
    def set_eval_results(self, metrics_dict):
        self.eval_metrics = metrics_dict

    # if we decide to include this function then
    # we have to do recursive update for nested dictionaries
    def update_eval_results(self, updated_metrics_dict):
        pass


    def print_loss_terms(self):
        str_key = (
            self.epochs[-1],
            self.total_loss_per_step[-1],
            self.nll_per_step[-1],
            self.reg_per_step[-1]
        )
        print('Epoch [%d]: total/nll/reg %.5f/%.5f/%.5f' %str_key)

