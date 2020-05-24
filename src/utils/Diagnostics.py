import numpy as np
import torch
from loss.loss_calculators import ExponentialLossCalculator
import sys
#from utils.loss_calculators import GGDLossCalculator
#from utils.loss_calculators import RegularizationCalculator
#from utils.loss_calculators import GammaLossCalculator
'''
Computes and holds model diagnostics during training
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


    def update(self,
        pred_params, hidden_states, total_loss,
        reg, logprob, epoch
    ):
        self.pred_params_per_step.append(pred_params)
        self.hidden_states_per_step.append(hidden_states)


        self.total_loss_per_step.append(total_loss)
        self.reg_per_step.append(reg)
        self.nll_per_step.append(-logprob)
        self.epochs.append(epoch)

    def print_loss_terms(self):
        str_key = (
            self.epochs[-1],
            self.total_loss_per_step[-1],
            self.nll_per_step[-1],
            self.reg_per_step[-1]
        )
        print('Epoch [%d]: total/nll/reg %.5f/%.5f/%.5f' %str_key)

