import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import gamma

class ResultsPlotterSynth:
    
    # assumes the learned parameters are in order by group with counts per group being the number of each group in order. 
    def __init__(self, model_results, model_type, plot_params):
        self.predicted_distribution_parameters = model_results['predicted_distribution_parameters']
        self.params = plot_params
        self.model_type = model_type
    
    def plot_learned_distribution_vs_true(self, true_parameters, counts_per_group):
        if self.model_type == 'exp':
            self.plot_exp_learned_vs_true_dist(true_parameters, counts_per_group)
            labels = ['rate']
        elif self.model_type == 'gamma':
            self.plot_gamma_learned_vs_true_dist(true_parameters, counts_per_group)
            labels = ['alpha', 'beta']
        elif self.model_type == 'ggd':
            self.plot_ggd_learned_vs_true_dist(true_parameters, counts_per_group)
            labels = ['alpha', 'beta', 'lambda']
        else:
            raise ValueError('Distribution type %s not found' %(self.model_type))

    def make_learned_vs_true_boxplot(self, true_parameters, labels=None):
        true_per_parameter_data = [true_parameters[i] for i in range(len(true_parameters))]
        #print(self.predicted_distribution_parameters)
        #print(true_parameters)
        pred_per_parameter_data = [[param[i] for param in self.predicted_distribution_parameters] for i in range(len(true_parameters[0]))]
        # so we have each true param boxplot next to the predicted one
        in_alternating_order = []
        doubled_labels = []
        for i in range(len(self.predicted_distribution_parameters[0])):
            in_alternating_order.append(true_per_parameter_data[i])
            in_alternating_order.append(pred_per_parameter_data[i])
            if labels:
                doubled_labels.append('True ' + labels[i])
                doubled_labels.append('Predicted ' + labels[i])
            
        
        plt.boxplot(in_alternating_order, showfliers=False, labels=doubled_labels if labels else None)
    
    def plot_exp_learned_vs_true_dist(self, true_parameters, counts_per_group, figscale=5):
        z = 1.96 # for plotting 95% confidence interval
        x_range = np.linspace(0, 5, 100)
        fig, axes = plt.subplots(len(counts_per_group), 1, figsize=(figscale * 1, figscale * 3))
        cur_idx = 0
        self.predicted_distribution_parameters = [self.predicted_distribution_parameters[i].detach().numpy() for i in range(len(self.predicted_distribution_parameters))]
        for group_idx, count in enumerate(counts_per_group):
            mean_pred_param = np.mean(self.predicted_distribution_parameters[cur_idx : count + cur_idx])
            std_pred_params = np.std(self.predicted_distribution_parameters[cur_idx: count + cur_idx])
            upper_confidence_interval = mean_pred_param + z * std_pred_params/(count**(.5))
            lower_confidence_interval = mean_pred_param - z * std_pred_params/(count**(.5))
            print(self.predicted_distribution_parameters[cur_idx : count + cur_idx])
            true_param = true_parameters[group_idx]
            print(mean_pred_param, true_param)
            pred_exp_pdf = mean_pred_param * np.exp(-mean_pred_param*x_range)
            true_exp_pdf = true_param * np.exp(-true_param * x_range)
            lower_exp_pdf = lower_confidence_interval * np.exp(-lower_confidence_interval * x_range)
            upper_exp_pdf = upper_confidence_interval * np.exp(-upper_confidence_interval * x_range)
            axes[group_idx].plot(x_range, pred_exp_pdf, label='Predicted Mean')
            axes[group_idx].plot(x_range, true_exp_pdf, label='True')
            axes[group_idx].plot(x_range, lower_exp_pdf, label='95% confidence interval, lower')
            axes[group_idx].plot(x_range, upper_exp_pdf, label='95% confidence interval, upper')
            axes[group_idx].legend()
            axes[group_idx].set_title('Mean Pred Param PDF for Group %d' %group_idx)

            
            cur_idx = count + cur_idx


    def plot_gamma_learned_vs_true_dist(self, true_parameters, counts_per_group, figscale=5):
        x_range = np.linspace(0, 5, 100)
        fig, axes = plt.subplots(len(counts_per_group), 1, figsize=(figscale * 1, figscale * 3))
        cur_idx = 0
        self.predicted_distribution_parameters = [self.predicted_distribution_parameters[i].detach().numpy() for i in range(len(self.predicted_distribution_parameters))]
        for group_idx, count in enumerate(counts_per_group):
            mean_pred_param = np.mean(self.predicted_distribution_parameters[cur_idx : count + cur_idx], axis=0)
            true_param = true_parameters[group_idx]
            print(mean_pred_param, true_param)
            pred_gamma_pdf = gamma.pdf(x_range, mean_pred_param[0], scale=1/mean_pred_param[1])
            true_gamma_pdf = gamma.pdf(x_range, true_param[0], scale=1/true_param[1])
            axes[group_idx].plot(x_range, pred_gamma_pdf, label='Predicted')
            axes[group_idx].plot(x_range, true_gamma_pdf, label='True')
            axes[group_idx].legend()
            axes[group_idx].set_title('Mean Pred Param PDF for Group %d' %group_idx)
            cur_idx = count + cur_idx


    def plot_ggd_learned_vs_true_dist(self, true_parameters):
        pass
