import sys
sys.path.append('../../data/')
import pickle
from make_simple_synth_data import SimpleData
from make_simple_synth_data import LearnedDataDeltaIJ
from make_simple_synth_data import LearnedDataThetaIJ
import matplotlib.pyplot as plt
from scipy.stats import gengamma
import numpy as np
import scipy.stats
from scipy.stats import gamma

class ResultsPlotterSynth:
    
    # assumes the learned parameters are in order by group with counts per group being the number of each group in order. 
    def __init__(self, learned_model, plot_params):
#        self.predicted_distribution_iparameters = pred_params 
        self.model = learned_model
        self.params = plot_params
        self.model_class_name = type(self.model).__name__
    
    def plot_learned_distribution_vs_true(self, true_parameters, counts_per_group):
        if self.model_type == 'exponential':
            self.plot_exp_learned_vs_true_dist(true_parameters, counts_per_group)
            labels = ['rate']
        elif self.model_type == 'weibull':
            pass
        else:
            raise ValueError('Distribution type %s not found' %(self.model_type))

    def make_learned_vs_true_boxplot(self, true_parameters, counts_per_group, labels=None, figsize=(50, 10)):
        plt.figure(figsize=figsize)
        pred_per_parameter_data = [[self.predicted_distribution_parameters[:, i]] for i in range(len(true_parameters[0]))]
        in_alternating_order = []
        doubled_labels = []
        cur_start_idx = 0
        for i, count in enumerate(counts_per_group):
            for param_idx in range(len(true_parameters[0])):
                true_param = [true_parameters[i][param_idx]]
                pred_params = [params[param_idx].detach().numpy() for params in self.predicted_distribution_parameters[cur_start_idx: cur_start_idx + count]]
                
                in_alternating_order.append(true_param)
                in_alternating_order.append(pred_params)
                #in_alternating_order.append(true_per_parameter_data[i])
                #in_alternating_order.append(pred_per_parameter_data[i])
                if labels:
                    doubled_labels.append('Group%d True ' %i + labels[param_idx])
                    doubled_labels.append('Group%d Predicted ' %i + labels[param_idx])
            cur_start_idx += count
            
        print(len(doubled_labels), len(in_alternating_order))
        print(in_alternating_order)
#        in_alternating_order = [[in_alternating_order[i][j].detach().numpy() for j in range(len(in_alternating_order[i]))] for i in range(len(in_alternating_order))]
        plt.boxplot(in_alternating_order, showfliers=False, labels=doubled_labels if labels else None)
    
    def plot_exp_learned_vs_true_dist(self, true_parameters, counts_per_group, figscale=5):
        true_parameters = [param[0] for param in true_parameters]
        z = 1.96 # for plotting 95% confidence interval
        x_range = np.linspace(0, 5, 100)
        fig, axes = plt.subplots(len(counts_per_group), 1, figsize=(figscale * 1, figscale * 3))
        cur_idx = 0
        predicted_distribution_parameters = [self.predicted_distribution_parameters[i].detach().numpy() for i in range(len(self.predicted_distribution_parameters))]
        for group_idx, count in enumerate(counts_per_group):
            mean_pred_param = np.mean(predicted_distribution_parameters[cur_idx : count + cur_idx])
            std_pred_params = np.std(predicted_distribution_parameters[cur_idx: count + cur_idx])
            #print(std_pred_params)
            upper_confidence_interval = mean_pred_param + z * std_pred_params/(count**(.5))
            lower_confidence_interval = mean_pred_param - z * std_pred_params/(count**(.5))
            #print(self.predicted_distribution_parameters[cur_idx : count + cur_idx])
            true_param = true_parameters[group_idx]
            #print(mean_pred_param, true_param)
            pred_exp_pdf = mean_pred_param * np.exp(-mean_pred_param*x_range)
            true_exp_pdf = true_param * np.exp(-true_param * x_range)
            lower_exp_pdf = lower_confidence_interval * np.exp(-lower_confidence_interval * x_range)
            upper_exp_pdf = upper_confidence_interval * np.exp(-upper_confidence_interval * x_range)
            axes[group_idx].plot(x_range, pred_exp_pdf, label='Predicted Mean')
            axes[group_idx].plot(x_range, true_exp_pdf, label='True')
            #axes[group_idx].plot(x_range, lower_exp_pdf, label='95% confidence interval, lower')
            #axes[group_idx].plot(x_range, upper_exp_pdf, label='95% confidence interval, upper')
            axes[group_idx].legend()
            axes[group_idx].set_title('Mean Pred PDF for Group %d' %group_idx)

            
            cur_idx = count + cur_idx


    def plot_gamma_learned_vs_true_dist(self, true_parameters, counts_per_group, figscale=5):
        # TODO set this range automatically around the true parameter
        x_range = np.linspace(0, 15, 100)
        fig, axes = plt.subplots(len(counts_per_group), 1, figsize=(figscale * 1, figscale * 3))
        cur_idx = 0
        self.predicted_distribution_parameters = [self.predicted_distribution_parameters[i].detach().numpy() for i in range(len(self.predicted_distribution_parameters))]
        for group_idx, count in enumerate(counts_per_group):
            mean_pred_param = np.mean(self.predicted_distribution_parameters[cur_idx : count + cur_idx], axis=0)
            print(mean_pred_param)
            true_param = true_parameters[group_idx]
            cur_preds = np.array(self.predicted_distribution_parameters[cur_idx: count + cur_idx])
            print(cur_preds[:, 0]/cur_preds[:, 1])
            print('Meant of ratio', np.mean(cur_preds[:, 0]/cur_preds[:, 1]))
            
            print('Variance of ratio', np.var(cur_preds[:, 0]/cur_preds[:, 1]))
            print(self.predicted_distribution_parameters[cur_idx: count + cur_idx])
            print(mean_pred_param, true_param)
            pred_gamma_pdf = gamma.pdf(x_range, mean_pred_param[0], scale=1/mean_pred_param[1])
            true_gamma_pdf = gamma.pdf(x_range, true_param[0], scale=1/true_param[1])
            axes[group_idx].plot(x_range, pred_gamma_pdf, label='Predicted')
            axes[group_idx].plot(x_range, true_gamma_pdf, label='True')
            axes[group_idx].legend()
            axes[group_idx].set_title('Mean Pred Param PDF for Group %d' %group_idx)
            cur_idx = count + cur_idx


    def plot_ggd_learned_vs_true_dist(self, true_parameters, counts_per_group, figscale=5):
        x_range = np.linspace(0, 15, 100)
        fig, axes = plt.subplots(len(counts_per_group), 1, figsize=(figscale * 1, figscale * 3))
        cur_idx = 0
        self.predicted_distribution_parameters = [self.predicted_distribution_parameters[i].detach().numpy() for i in range(len(self.predicted_distribution_parameters))]
        for group_idx, count in enumerate(counts_per_group):
            mean_pred_param = np.mean(self.predicted_distribution_parameters[cur_idx : count + cur_idx], axis=0)
            true_param = true_parameters[group_idx]
            cur_preds = np.array(self.predicted_distribution_parameters[cur_idx: count + cur_idx])
            
            #print(self.predicted_distribution_parameters[cur_idx: count + cur_idx])
            print(mean_pred_param, true_param)
            pred_gamma_pdf = gengamma.pdf(x_range, mean_pred_param[2], mean_pred_param[0], loc=mean_pred_param[1])
            true_gamma_pdf = gengamma.pdf(x_range, true_param[0], true_param[1],  loc=true_param[2])
            axes[group_idx].plot(x_range, pred_gamma_pdf, label='Predicted')
            axes[group_idx].plot(x_range, true_gamma_pdf, label='True')
            axes[group_idx].legend()
            axes[group_idx].set_title('Mean Pred Param PDF for Group %d' %group_idx)
            cur_idx = count + cur_idx
    
    def plot_event_time_samples_from_learned_model(self, path_to_synth_data, savepath):
        model_class_name = self.model_class_name
        if model_class_name[0:6] == 'Linear':
            plot_func = self.plot_event_time_samples_from_linear_model
        else:
            error = 'Plotting for model type %s not yet implemented' %model_class_name
            raise NotImplementedError(error)
        plot_func(path_to_synth_data, savepath)

    def plot_event_time_samples_from_linear_model(self, 
        path_to_synth_data, savepath, figsize=(5, 10)
    ):
        learned_model = self.model
        with open(path_to_synth_data, 'rb') as f:
            synth_true_data = pickle.load(f)
        # uses learned model parameters with same poisson rates, and cov trajs
        # in order to sample from the learned data distribution
        if synth_true_data.dataset_type == 'delta_per_step':
            data_sampled_from_model = LearnedDataDeltaIJ(learned_model, synth_true_data)
        elif synth_true_data.dataset_type == 'theta_per_step':
            data_sampled_from_model = LearnedDataThetaIJ(learned_model, synth_true_data)
        else:
            raise ValueError('Dataset type %s not recognized' %(synth_true_data.dataset_type))
        data_sampled_from_model.construct_data()
        print('Data sampled from model!')

        fig, axes = plt.subplots(
            2, 1, figsize=figsize,
            sharex=True, sharey=True
        )
        axes[0].set_title('Data Sampled from Model')
        axes[1].set_title('True Data')
        self.plot_data_event_times_hist(synth_true_data, axes[1])
        print('Synth plotted!')
        self.plot_data_event_times_hist(data_sampled_from_model, axes[0])
        print('Learned model samples plotted!')
        plt.savefig(savepath)



    def plot_data_event_times_hist(self, learned_data, axis):
        n_per_class = learned_data.n_per_class
        axis.hist(
            learned_data.true_event_times[0:n_per_class[0]],
            alpha=.5, label='unhealthy', bins=50, density=True
        )
        axis.hist(
            learned_data.true_event_times[n_per_class[0]:],
            alpha=.5, label='healthy', bins=50, density=True
        )
        axis.legend()
                    
