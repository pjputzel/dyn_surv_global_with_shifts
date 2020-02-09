import numpy as np
import torch


'''
To add a new metric just include the function for computing that metric,
and add it to self.supported_metrics_funcs
'''
class MetricsTracker:
    def __init__(self, model, data_input, metric_names):
        self.data_input = data_input
        self.metric_names = metric_names
        self.model = model
        self.epochs = []
        self.supported_metrics_funcs =\
            {
               'dummy_test_function': self.dummy_test_function 
            }
        self.init_metric_funcs_and_metrics(metric_names)
    
    def init_metric_funcs_and_metrics(self, metric_names):
        self.metric_funcs = {}
        self.metrics = {}
        for metric_name in metric_names:
            if metric_name in self.supported_metrics_funcs:
                self.metric_funcs[metric_name] = self.supported_metrics_funcs[metric_name]
                self.metrics[metric_name] = []
            else:
                raise ValueError('Metric %s not implemented' %metric_name)

    def update(self, epoch, model_outputs_per_batch_tr, model_outputs_per_batch_te=None):
        self.epochs.append(epoch)
        if model_outputs_per_batch_te:
            cur_outputs = {'tr': model_outputs_per_batch_tr, 'te': model_outputs_per_batch_te}
        else:
            cur_outputs = {'tr': model_outputs_per_batch_tr}
        for metric_name in self.metric_names:
            self.metrics[metric_name].append(self.metric_funcs[metric_name](cur_outputs))
    

    def dummy_test_function(self, cur_outputs):
        print('meow')
