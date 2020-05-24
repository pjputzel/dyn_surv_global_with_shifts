import numpy as np
import torch
from loss.loss_calculators import ExponentialLossCalculator
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


    def compute_batch_diagnostics(self, model, cur_batch, cur_batch_event_times, cur_batch_censoring_indicators):
        
        unpacked_cur_batch, lengths = torch.nn.utils.rnn.pad_packed_sequence(cur_batch)
        unpacked_cur_batch_cov = unpacked_cur_batch.permute(1, 0, 2)
        #cur_batch_event_time_deltas = cur_batch_event_times.unsqueeze(1) - unpacked_cur_batch[:, :, 0]
        #assert(torch.sum(cur_batch_event_time_deltas < 0) == 0) 
 
        cur_diagnostics = {}
        cur_diagnostics['sequence_lengths'] = lengths
        cur_diagnostics['next_step_cov_preds'], cur_diagnostics['predicted_distribution_params'], cur_diagnostics['lengths'] = model(cur_batch)
        cur_diagnostics['loss'] = self.loss_calculator.compute_loss(cur_batch_event_times, unpacked_cur_batch_cov[:, :, 0], lengths, cur_diagnostics['predicted_distribution_params'], cur_batch_censoring_indicators, one_theta=self.one_theta)
        cur_diagnostics['regularization'] = self.regularization_calculator.compute_regularization(cur_batch, cur_diagnostics)
        # TODO: debug nans in regularization!!
        cur_diagnostics['total_loss'] = cur_diagnostics['loss'] + cur_diagnostics['regularization'] 
        #cur_diagnostics['regularization'] = self.regularization_calculator.compute_regularization(cur_batch, cur_diagnostics)
        self.cur_diagnostics = cur_diagnostics
        return cur_diagnostics 

    def update_full_data_diagnostics(self, diagnostics_per_batch, epoch):
        full_data_diagnostics = {}
        # now aggregate the batch diagnostics
        full_data_diagnostics['loss'] = 1/(len(diagnostics_per_batch)) * torch.sum(torch.tensor([diag['loss'] for diag in diagnostics_per_batch]))

        #full_data_diagnostics['next_step_cov_preds'] = torch.cat([cur_diag['next_step_cov_preds'] for cur_diag in diagnostics_per_batch], axis=1)
        full_data_diagnostics['predicted_distribution_params'] = [cur_diag['predicted_distribution_params'] for cur_diag in diagnostics_per_batch]

        full_data_diagnostics['regularization'] = 1/(len(diagnostics_per_batch)) * torch.sum(torch.tensor([diag['regularization'] for diag in diagnostics_per_batch]))
        # TODO: fix bug in regularization and uncomment
        full_data_diagnostics['total_loss'] = full_data_diagnostics['loss'] + full_data_diagnostics['regularization']
        self.full_data_diagnostics = full_data_diagnostics
        self.epochs.append(epoch)
        return full_data_diagnostics

    def print_cur_diagnostics(self):
        print('total/nll/reg %.5f/%.5f/%.5f' %(self.full_data_diagnostics['total_loss'] ,self.full_data_diagnostics['loss'], self.full_data_diagnostics['regularization']))

    def unshuffle_results(self, unshuffle_idxs):
        for key, value in self.full_data_diagnostics.items():
            if key in ['loss', 'regularization', 'total_loss', 'epochs']:
                continue
            all_values = torch.cat(value)
            self.full_data_diagnostics[key] = [all_values[unshuffle_idx] for unshuffle_idx in unshuffle_idxs]
            #self.cur_diagnostics[key] = [self.cur_diagnostics[key][unshuffle_idx] for unshuffle_idx in unshuffle_idxs] 
        
