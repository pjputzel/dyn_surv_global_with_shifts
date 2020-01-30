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
                'tr_acc': self.compute_tr_acc, 'te_acc': self.compute_te_acc,
                'tr_auc': self.compute_tr_auc, 'te_auc': self.compute_te_auc,
                'tr_log_loss': self.compute_tr_log_loss, 
                'te_log_loss': self.compute_te_log_loss,
                'tr_features': self.compute_tr_features,
                'te_features': self.compute_te_features,
                'tr_avg_pos_feat': self.compute_tr_pos_feat_avg,
                'te_avg_pos_feat': self.compute_te_pos_feat_avg,
                'tr_avg_neg_feat': self.compute_tr_neg_feat_avg,
                'te_avg_neg_feat': self.compute_te_neg_feat_avg,
                'tr_reg_feat_diff': self.compute_tr_reg_feat_diff,
                'te_reg_feat_diff': self.compute_te_reg_feat_diff,
                'tr_feat_diff_reg_per_sample': self.compute_tr_reg_feat_diff_per_sample,
                'dummy_test_function': self.meow
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

    def update(self, epoch):
        self.epochs.append(epoch)
        output_tr = self.model(self.data_input.x_tr, self.data_input.y_tr)
        self.add_feat_diff_reg_per_sample_to_tr_output(output_tr) 
        output_te = self.model(self.data_input.x_te, self.data_input.y_te)
        cur_outputs = {'tr': output_tr, 'te': output_te}
        for metric_name in self.metric_names:
            self.metrics[metric_name].append(self.metric_funcs[metric_name](cur_outputs))
    
    def add_feat_diff_reg_per_sample_to_tr_output(self, cur_outputs_tr):
        pos_mean = self.compute_pos_mean_feat(cur_outputs_tr)
        neg_mean = self.compute_neg_mean_feat(cur_outputs_tr)
        estimated_feat_diff_regs_per_sample = []
        for i, label in enumerate(self.data_input.y_tr):
            if label == 0:
                estimated_feat_diff_regs_per_sample.append(self.get_feat_diff_reg(cur_outputs_tr['leaf_logp'][i], pos_mean))
            else:
                estimated_feat_diff_regs_per_sample.append(self.get_feat_diff_reg(cur_outputs_tr['leaf_logp'][i], neg_mean))
        
        cur_outputs_tr['feat_diff_reg_per_sample'] = estimated_feat_diff_regs_per_sample
        
    def compute_pos_mean_feat(self, cur_outputs):
        return np.mean(cur_outputs['leaf_logp'][self.data_input.y_tr == 1].detach().numpy())

    def compute_neg_mean_feat(self, cur_outputs):
        return np.mean(cur_outputs['leaf_logp'][self.data_input.y_tr == 0].detach().numpy())

    def get_feat_diff_reg(self, feature, relative_mean):
        return -np.log(((feature - relative_mean)**2).detach().numpy())

    def compute_tr_acc(self, cur_outputs):
        tr_output = cur_outputs['tr']
        y_true = self.data_input.y_tr
        y_pred = (tr_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
        y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
        acc = sum(y_pred == y_true.cpu().numpy()) * 1.0 / y_true.shape[0]
        return acc

    def compute_te_acc(self, cur_outputs):
        te_output = cur_outputs['te']
        y_true = self.data_input.y_te
        y_pred = (te_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
        y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
        acc = sum(y_pred == y_true.cpu().numpy()) * 1.0 / y_true.shape[0]
        return acc

    def compute_tr_auc(self, cur_outputs):
        tr_output = cur_outputs['tr']
        y_pred = (tr_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
        y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
        y_true = self.data_input.y_tr
        return roc_auc_score(y_true.cpu().detach().numpy(), y_pred, average='macro')

    def compute_te_auc(self, cur_outputs):
        te_output = cur_outputs['te']
        y_pred = (te_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
        y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
        y_true = self.data_input.y_te
        return roc_auc_score(y_true.cpu().detach().numpy(), y_pred, average='macro')
    
    def compute_tr_features(self, cur_outputs):
        return self.compute_features(cur_outputs, split='tr')

    def compute_te_features(self, cur_outputs):
        return self.compute_features(cur_outputs, split='te')

    def compute_tr_log_loss(self, cur_outputs):
        return cur_outputs['tr']['log_loss']

    def compute_te_log_loss(self, cur_outputs):
        return cur_outputs['te']['log_loss']

    def compute_tr_reg_feat_diff(self, cur_outputs):
        return self.compute_feature_diff_reg(cur_outputs, split='tr')

    def compute_te_reg_feat_diff(self, cur_outputs):
        return self.compute_feature_diff_reg(cur_outputs, split='te')

    def compute_feature_diff_reg(self, cur_outputs, split='tr'):
        return cur_outputs[split]['feature_diff_reg']

    def compute_neg_prop_reg(self, cur_outputs, split='tr'):
        return cur_outputs[split]['emp_reg_loss']

    def compute_tr_pos_feat_avg(self, cur_outputs):
        features = cur_outputs['tr']['leaf_logp']
        pos_features = [feature.cpu().detach().numpy() for i, feature in enumerate(features) if self.data_input.y_tr[i] == 1]
        return np.mean(pos_features)

    def compute_te_pos_feat_avg(self, cur_outputs):
        features = cur_outputs['te']['leaf_logp']
        pos_features = [feature.cpu().detach().numpy() for i, feature in enumerate(features) if self.data_input.y_te[i] == 1]
        return np.mean(pos_features)

    def compute_tr_neg_feat_avg(self, cur_outputs):
        features = cur_outputs['tr']['leaf_logp']
        neg_features = [feature.cpu().detach().numpy() for i, feature in enumerate(features) if self.data_input.y_tr[i] == 0]
        return np.mean(neg_features)

    def compute_te_neg_feat_avg(self, cur_outputs):
        features = cur_outputs['te']['leaf_logp']
        neg_features = [feature.cpu().detach().numpy() for i, feature in enumerate(features) if self.data_input.y_te[i] == 0]
        return np.mean(neg_features)


    def compute_features(self, cur_outputs, split='tr'):
        return cur_outputs[split]['leaf_logp']

    def compute_tr_reg_feat_diff_per_sample(self, cur_outputs):
        return cur_outputs['tr']['feat_diff_reg_per_sample']
    
    def meow(self, cur_outputs):
        return 'meow'
