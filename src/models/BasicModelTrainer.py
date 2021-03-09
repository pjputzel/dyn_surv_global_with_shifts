import torch
import pickle
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
        if self.params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['loss_params']['l2_reg'])
        elif self.params['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.params['learning_rate'], weight_decay=self.params['loss_params']['l2_reg'])

        else:
            raise ValueError('optimizer type %s not recognized' %self.params['optimizer'])
        
        #self.lr_scheduler = torch.optim.lr_scheduler.steplr(self.optimizer, self.params['max_iter']/10)
        #self.lr_scheduler = torch.optim.lr_scheduler.cosineannealingwarmrestarts(self.optimizer, int(self.params['max_iter']/10))
        while torch.abs(cur_loss - prev_loss) > self.params['conv_thresh'] and epoch < self.params['max_iter']:
            data_input.make_randomized_tr_batches(self.params['batch_size'])

            total_loss, reg, logprob, grad_mag =\
                self.step_params_over_all_batches(model, data_input)

            self.diagnostics.update(
                total_loss, reg, logprob, epoch,
                grad_mag
            )
            if epoch % self.params['n_epoch_print'] == 0:
                if self.metric_evaluator and self.params['track_c_indices']:
                    self.compute_cur_tracked_metrics(model, data_input)
                self.diagnostics.print_loss_terms()


            #self.lr_scheduler.step()
            prev_loss = cur_loss
            cur_loss = total_loss
            epoch += 1

        # update one last time
        self.diagnostics.update(
            total_loss, reg, logprob, epoch,
            grad_mag
        )
        self.diagnostics.print_loss_terms()
        print('Total training time was %d seconds'\
            %(time.time() - start_time)
        )
        with open('extreme_loss_counts.pkl', 'wb') as f:
            pickle.dump(self.loss_calc.logprob_calculator.extreme_loss_counts, f)
        
        return self.diagnostics
#    @profile
    def step_params_over_all_batches(self, model, data_input):
        pred_params_per_batch, hidden_states_per_batch, step_ahead_cov_preds_per_batch = [], [], []
        grad_mag_per_batch = []
        total_loss_per_batch, reg_per_batch, logprob_per_batch = [], [], []
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

            grad_mag_per_batch.append(self.get_grad_magnitude(model))
            total_loss_per_batch.append(total_loss)
            reg_per_batch.append(reg)
            logprob_per_batch.append(logprob)
        total_loss_avg = torch.mean(torch.tensor(total_loss_per_batch))
        reg_avg = torch.mean(torch.tensor(reg_per_batch))
        logprob_avg = torch.mean(torch.tensor(logprob_per_batch))
        grad_mag_avg = torch.mean(torch.tensor(grad_mag_per_batch))
        return total_loss_avg, reg_avg, logprob_avg, grad_mag_avg

    def get_grad_magnitude(self, model):
        grad_mag_sq = 0
        for name, param in model.named_parameters():
            # check for none with params which aren't used in current code
            # probably should just remove these params
            if param.requires_grad and not param.grad is None:
                #print(param, param.grad is None, param.shape, name)
                param_mag = torch.sum(param.grad**2)
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
        self.diagnostics.update_tracked_eval_metrics()



