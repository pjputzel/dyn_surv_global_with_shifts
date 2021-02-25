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
        total_loss, reg, logprob, epoch,
        grad_mag
        #pred_params, hidden_states, total_loss,
    ):
#        self.pred_params_per_step.append(pred_params.cpu().detach())
#        if not ignore_hidden_states:
#            self.hidden_states_per_step.append(hidden_states)
#        else:
#        self.hidden_states_per_step.append(hidden_states.cpu().detach())

        #print([sys.getsizeof(self.hidden_states_per_step[i].storage()) for i in range(len(self.hidden_states_per_step))])
        #print([sys.getsizeof(self.pred_params_per_step[i].storage()) for i in range(len(self.pred_params_per_step))])
        if hasattr(self, 'cur_tracked_eval_metrics'):
            self.update_tracked_eval_metrics()

        self.total_loss_per_step.append(total_loss.cpu().detach().numpy())
        self.reg_per_step.append(0 if type(reg) is float else reg.cpu().detach().numpy())
        self.nll_per_step.append(-logprob.cpu().detach().numpy())
        self.epochs.append(epoch)
        self.grad_magnitude_per_step.append(grad_mag.cpu().detach().numpy())

    
    def set_eval_results(self, metrics_dict):
        self.eval_metrics = metrics_dict

    # if we decide to include this function then
    # we have to do recursive update for nested dictionaries
    def update_eval_results(self, updated_metrics_dict):
        pass

    def update_tracked_eval_metrics(self):
        metric_names = list(self.cur_tracked_eval_metrics.keys())
        for metric_name in metric_names:
            metric_res_name_tr = metric_name + '_tracking_tr'
            metric_res_name_te = metric_name + '_tracking_te'
            if not hasattr(self, metric_res_name_tr):
                setattr(self, metric_res_name_tr, [])
                setattr(self, metric_res_name_te, [])
            tr_mean = np.mean(
                self.cur_tracked_eval_metrics[metric_name]['tr']['values'].cpu().detach().numpy()
            )
            te_mean = np.mean(
                self.cur_tracked_eval_metrics[metric_name]['te']['values'].cpu().detach().numpy()
            )
            self.__dict__[metric_res_name_tr].append(tr_mean)
            self.__dict__[metric_res_name_te].append(te_mean)

    def print_loss_terms(self):
        str_key = (
            self.epochs[-1],
            self.total_loss_per_step[-1],
            self.nll_per_step[-1],
            self.reg_per_step[-1]
        )
        print('Epoch [%d]: total/nll/reg %.5f/%.5f/%.5f' %str_key)

