import matplotlib.pyplot as plt
import numpy as np

class ResultsPlotterSynth:
    
    # assumes the learned parameters are in order by group with counts per group being the number of each group in order. 
    def __init__(self, model_results, model_type, plot_params):
        self.predicted_distribution_parameters = model_results['predicted_distribution_parameters']
        self.params = plot_params
        self.model_type = model_type
    
    def plot_learned_distribution_vs_true(self, true_parameters, counts_per_group):
        if self.model_type == 'exp':
            self.plot_exp_learned_vs_true_dist(true_parameters, counts_per_group) 
        elif self.model_type == 'ggd':
            self.plot_ggd_learned_vs_true_dist(true_parameters, counts_per_group)
        else:
            raise ValueError('Distribution type %s not found' %(self.model_type))

    def plot_exp_learned_vs_true_dist(self, true_parameters, counts_per_group, figscale=5):
        x_range = np.linspace(0, 5, 100)
        fig, axes = plt.subplots(len(counts_per_group), 1, figsize=(figscale * 1, figscale * 3))
        cur_idx = 0
        for group_idx, count in enumerate(counts_per_group):
            mean_pred_param = np.mean(self.predicted_distribution_parameters[cur_idx : count + cur_idx].detach().numpy())
            true_param = true_parameters[group_idx]
            print(mean_pred_param, true_param)
            pred_exp_pdf = mean_pred_param * np.exp(-mean_pred_param*x_range)
            true_exp_pdf = true_param * np.exp(-true_param * x_range)
            axes[group_idx].plot(x_range, pred_exp_pdf, label='Predicted')
            axes[group_idx].plot(x_range, true_exp_pdf, label='True')
            axes[group_idx].legend()
            axes[group_idx].set_title('Mean Pred Param PDF for Group %d' %group_idx)
            cur_idx = count + cur_idx


    def plot_ggd_learned_vs_true_dist(self, true_parameters):
        pass
