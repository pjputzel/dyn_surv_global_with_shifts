import torch
import time
import math
import numpy as np
import time
from utils.Diagnostics import Diagnostics
from loss.LossCalculator import LossCalculator
import sys

class BasicModelTrainer:
    def __init__(self, train_params, model_type, metric_evaluator=None):
        self.params = train_params
        self.diagnostics = Diagnostics(train_params['diagnostic_params'])
        self.loss_calc = LossCalculator(train_params['loss_params'], model_type)
        self.metric_evaluator = metric_evaluator

#    @profile
    def train_model(self, model, data_input):
        self.diagnostics.padding_indicators = data_input.padding_indicators
        start_time = time.time()
        prev_loss = torch.tensor(0)
        cur_loss = torch.tensor(np.inf)
        epoch = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'])
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.params['max_iter']/10)
        #self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, int(self.params['max_iter']/10))
        while torch.abs(cur_loss - prev_loss) > self.params['conv_thresh'] and epoch < self.params['max_iter']:
            data_input.make_randomized_tr_batches(self.params['batch_size'])

            #pred_params, hidden_states
            total_loss, reg, logprob, grad_mag =\
                self.step_params_over_all_batches(model, data_input)

            if epoch % self.params['n_epoch_print'] == 0:
                if self.metric_evaluator:
                    self.compute_cur_tracked_metrics(model, data_input)
                self.diagnostics.update(
                    total_loss, reg, logprob, epoch,
                    grad_mag
                    #pred_params, hidden_states, 
                )
                self.diagnostics.print_loss_terms()
            #self.lr_scheduler.step()
            prev_loss = cur_loss
            cur_loss = total_loss
            epoch += 1

        # update one last time
        self.diagnostics.update(
            total_loss, reg, logprob, epoch,
            grad_mag
            #pred_params, hidden_states, 
        )
        self.diagnostics.print_loss_terms()
        print('Total training time was %d seconds'\
            %(time.time() - start_time)
        )
        # print out the learned params and compare to cov times
        
#        print(torch.cat([torch.exp(-self.diagnostics.pred_params_per_step[-1]), torch.max(data_input.cov_times, dim=1)[0].unsqueeze(1)], dim=1))
        #print(model.get_global_param())
        return self.diagnostics
#    @profile
    def step_params_over_all_batches(self, model, data_input):
        pred_params_per_batch, hidden_states_per_batch, step_ahead_cov_preds_per_batch = [], [], []
        grad_mag_per_batch = []
        total_loss_per_batch, reg_per_batch, logprob_per_batch = [], [], []
#        print('len of data_input.tr_batches %d' %len([b for b in data_input.tr_batches]))
        for batch in data_input.tr_batches:
            self.optimizer.zero_grad()

            pred_params, hidden_states, step_ahead_cov_preds = model(batch)
                
            total_loss, reg, logprob =\
                self.loss_calc.compute_batch_loss(
                    model, pred_params, hidden_states, 
                    step_ahead_cov_preds, batch
                )
            total_loss.backward()            
            self.optimizer.step()

#            pred_params_per_batch.append(pred_params)
#            hidden_states_per_batch.append(hidden_states)
            #step_ahead_cov_preds_per_batch.append(step_ahead_cov_preds)
            grad_mag_per_batch.append(self.get_grad_magnitude(model))
            total_loss_per_batch.append(total_loss)
            reg_per_batch.append(reg)
            logprob_per_batch.append(logprob)
        total_loss_avg = torch.mean(torch.tensor(total_loss_per_batch))
        reg_avg = torch.mean(torch.tensor(reg_per_batch))
        logprob_avg = torch.mean(torch.tensor(logprob_per_batch))
        grad_mag_avg = torch.mean(torch.tensor(grad_mag_per_batch))
        # combine and unshuffle to get *_all stuff
#        pred_params_all, hidden_states_all, total_loss_avg, reg_avg, logprob_avg = \
#            self.combine_batch_results(
#                data_input.unshuffled_tr_idxs,
#                pred_params_per_batch, hidden_states_per_batch, total_loss_per_batch,
#                reg_per_batch, logprob_per_batch
#            )        
#        return pred_params_all, hidden_states_all, total_loss_avg, reg_avg, logprob_avg
        return total_loss_avg, reg_avg, logprob_avg, grad_mag_avg

    def get_grad_magnitude(self, model):
        grad_mag_sq = 0
        for param in model.parameters():
            param_mag = torch.sum(param**2)
            grad_mag_sq += param_mag
        return grad_mag_sq ** (1/2)

    def compute_cur_tracked_metrics(self, model, data_input):
        device = next(model.parameters()).device
        if next(model.parameters()).is_cuda:
            model.to('cpu')
            data_input.to_device('cpu')
        model.eval()
        self.metric_evaluator.evaluate_model(
            model, data_input,
            self.diagnostics, is_during_training=True
        )
        model.train()
        if not device == 'cpu':
            model.to(device)
            data_input.to_device(device) 


#    def combine_batch_results(self, 
#        unshuffled_idxs,
#        pred_params_per_batch, hidden_states_per_batch, total_loss_per_batch,
#        reg_per_batch, logprob_per_batch
#    ):
#        if len(pred_params_per_batch) == 1:
#            # only one batch, no concatenation/averaging needed
#            return pred_params_per_batch[0][unshuffled_idxs],\
#                hidden_states_per_batch[0][unshuffled_idxs],\
#                total_loss_per_batch[0], reg_per_batch[0], logprob_per_batch[0]
#    
#        pred_params_all = torch.cat(pred_params_per_batch)
#        pred_params_all = pred_params_all[unshuffled_idxs]
#        
#        hidden_states_all = torch.cat(hidden_states_per_batch)
#        hidden_states_all = hidden_states_all[unshuffled_idxs]
#
#        total_loss_avg = torch.mean(torch.tensor(total_loss_per_batch))
#        reg_avg = torch.mean(torch.tensor(reg_per_batch))
#        logprob_avg = torch.mean(torch.tensor(logprob_per_batch))
#        
#        return pred_params_all, hidden_states_all, total_loss_avg, reg_avg, logprob_avg


# simple helper class that formats and stores the results from a single step
# maybe should be in same file as diagnostics class
#class ResultsSingleStep:
#    
#    def __init__(self):
#        self.pred_params_per_batch = []
#        self.hidden_states_per_batch = []
#        self.step_ahead_cov_preds_per_batch = []
#        self.total_loss_per_batch = []
#        self.reg_per_batch = []
#        self.logprob_per_batch = []
#        self.unshuffled_idxs_per_batch = []
#
#    def update(self, pred_params, hidden_states, step_ahead_cov_preds,
#        total_loss, reg, logprob, unshuffled_idxs):
#
#        self.pred_params_per_batch.append(pred_params)
#        self.hidden_states_per_batch.append(hidden_states)
#        self.step_ahead_cov_preds_per_batch.append(step_ahead_cov_preds)
#        self.total_loss_per_batch.append(total_loss)
#        self.reg_per_batch.append(reg)
#        self.logprob_per_batch.append(logprob)
#        self.unshuffled_idxs_per_batch.append(unshuffled_idxs)
#
#    def combine_and_format_results(self):
#        self.pred_params_all = torch.cat(self.pred_params_per_batch)
#        # TODO: update this based on what the model output actually is
