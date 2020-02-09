import numpy as np
import torch
from utils.loss_calculators import ExponentialLossCalculator
from utils.loss_calculators import RegularizationCalculator
'''
Computes and holds model diagnostics during training
'''

class Diagnostics:

    def __init__(self, diagnostic_params):
        self.params = diagnostic_params
        self.init_loss_calculators()
        self.epochs = []

    def init_loss_calculators(self):
        if self.params['distribution_type'] == 'exp':
            self.loss_calculator = ExponentialLossCalculator()
        elif self.params['distribution_type'] == 'ggd':
            self.loss_calculator = GGDLossCalculator()
        else:
            raise ValueError('Loss type not found')
        self.regularization_calculator = RegularizationCalculator(self.params['regularization_params'])

    def compute_batch_diagnostics(self, model, cur_batch, cur_batch_event_times, cur_batch_censoring_indicators):
        cur_diagnostics = {}
        cur_diagnostics['next_step_cov_preds'], cur_diagnostics['predicted_distribution_params'] = model(cur_batch)
        cur_diagnostics['loss'] = self.loss_calculator.compute_loss(cur_batch_event_times, cur_diagnostics['predicted_distribution_params'], cur_batch_censoring_indicators)
        cur_diagnostics['total_loss'] = cur_diagnostics['loss'] + self.regularization_calculator.compute_regularization(cur_batch, cur_diagnostics)
        cur_diagnostics['regularization'] = self.regularization_calculator.compute_regularization(cur_batch, cur_diagnostics)
        self.cur_diagnostics = cur_diagnostics
        return cur_diagnostics 

    def update_full_data_diagnostics(self, diagnostics_per_batch, epoch):
        full_data_diagnostics = {}
        # now aggregate the batch diagnostics
        full_data_diagnostics['loss'] = 1/(len(diagnostics_per_batch)) * torch.sum(torch.tensor([diag['loss'] for diag in diagnostics_per_batch]))

        #full_data_diagnostics['next_step_cov_preds'] = torch.cat([cur_diag['next_step_cov_preds'] for cur_diag in diagnostics_per_batch], axis=1)

        full_data_diagnostics['predicted_distribution_params'] = torch.cat([cur_diag['predicted_distribution_params'] for cur_diag in diagnostics_per_batch])

        full_data_diagnostics['regularization'] = 1/(len(diagnostics_per_batch)) * torch.sum(torch.tensor([diag['regularization'] for diag in diagnostics_per_batch]))

        full_data_diagnostics['total_loss'] = full_data_diagnostics['loss'] + full_data_diagnostics['regularization']
        self.full_data_diagnostics = full_data_diagnostics
        self.epochs.append(epoch)
        return full_data_diagnostics

    def print_cur_diagnostics(self):
        print('total loss/nll/reg %.5f/%.5f/%.5f' %(self.full_data_diagnostics['total_loss'] ,self.full_data_diagnostics['loss'], self.full_data_diagnostics['regularization']))
        
