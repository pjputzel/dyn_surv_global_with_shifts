import numpy
import torch
import matplotlib.pyplot as plt
import os


COLORS = {
    'dummy_global': 'r', 
    'linear_theta_per_step': 'b',
    'linear_delta_per_step': 'g'
}

class DynamicMetricsPlotter:

    def __init__(self, plot_params, savedir):
        self.params = plot_params
        if not os.path.exists(os.path.join(savedir, 'plots')):
            os.makedirs(os.path.join(savedir, 'plots'))
        self.savedir = os.path.join(savedir, 'plots')

    def make_and_save_dynamic_eval_metrics_with_bins_plots(self, eval_metrics_res):
        eval_metric_names = eval_metrics_res.keys()
        for metric_name in eval_metric_names:
            for split in eval_metrics_res[metric_name].keys():
                save_name_prefix = metric_name +  '_' + split
                # values array has the eval metrics result and has shape
                # (n_start_times, n_groups, n_time_deltas)
                num_groups = eval_metrics_res[metric_name][split]['values'].shape[1]
                for group in range(num_groups):
                    # get the modified version of the dictionary just for this group
                    metrics_res_cur = eval_metrics_res[metric_name][split]
                    single_group_eval_metric_res = {\
                        'start_times': metrics_res_cur['start_times'],
                        'time_deltas': metrics_res_cur['time_deltas'],
                        'values': metrics_res_cur['values'][:, group, :],
                        'eff_n': \
                            [   eff_ns_s[group]
                                for eff_ns_s in metrics_res_cur['eff_n']
                            ]
                    }
                    end_bin = metrics_res_cur['bin_upper_boundaries'][group]
                    start_bin = metrics_res_cur['bin_upper_boundaries'][group - 1]
                    if group == 0:
                        start_bin = 0
                    bin_start_and_finish = '%d events to %d events' %(start_bin, end_bin)
                    save_name = save_name + bin_start_and_finish + '.png'
                    self.make_and_save_single_metric_plots(
                        single_group_eval_metric_res,
                        save_name
                    )
        




    def make_and_save_dynamic_eval_metrics_plots(self, eval_metrics_res):
        eval_metric_names = eval_metrics_res.keys()
        for metric_name in eval_metric_names:
            for split in eval_metrics_res[metric_name].keys():
                save_name = metric_name +  '_' + split + '.png'
                self.make_and_save_single_metric_plots(
                    eval_metrics_res[metric_name][split],
                    save_name
                )

    def make_and_save_single_metric_plots(self, metric_res, save_name):
        metric_name = save_name.split('_')[0]
        start_times = metric_res['start_times']
        time_deltas = metric_res['time_deltas']
        values = metric_res['values']
        eff_n = metric_res['eff_n']

        for s, start in enumerate(start_times):
            fig, axes = plt.subplots(2, 1, figsize=(5, 10))
            fig.suptitle(save_name.split('.')[0] + '_S=%.2f' %start)
            self.plot_eff_n_vs_deltas(
                axes[0], save_name, eff_n, 
                s, time_deltas
            )
            self.plot_metric_values_vs_deltas(
                axes[1], save_name, values, 
                s, time_deltas
            )
            save_path = os.path.join(self.savedir, save_name.split('.')[0] + '_S=%s.png' %start)
            plt.savefig(save_path)


    def plot_eff_n_vs_deltas(self, 
        axis, save_name, eff_n, 
        s, time_deltas
    ):
        metric_name = save_name.split('_')[0]
        if metric_name == 'auc':
            #axis.set_title('Cases and Controls Vs Elapsed Time')
            case_ns = [eff_n[s][i]['cases'] for i in range(len(eff_n[s]))]
            control_ns = [eff_n[s][i]['controls'] for i in range(len(eff_n[s]))]
            axis.plot(time_deltas, case_ns, label='Cases')
            axis.plot(time_deltas, control_ns, label='Controls')
            axis.legend()
            axis.set_ylabel('Number of Individuals')
        else:
            axis.set_title('Number of Valid Pairs vs Elapsed Time')
            axis.plot(time_deltas, eff_n[s])
            axis.set_ylabel('Number of Valid Pairs')
        axis.set_xlabel('Time Elapsed')
        
    def plot_metric_values_vs_deltas(self, 
        axis, save_name, values, 
        s, time_deltas, model_name=None
    ):
        metric_name = save_name.split('_')[0]
        axis.set_ylabel(metric_name.title())
        axis.set_xlabel('Time Elapsed')
        if model_name:
            axis.scatter(time_deltas, values[s, :], c=COLORS[model_name])
        else:
            axis.scatter(time_deltas, values[s, :])
            
        
        

               
    # TODO: implement the following
    # should take in a list of eval_metrics_results per_model 
    # output the plots with all models on the metric and eff_n at the top
    def make_and_save_dynamic_eval_metrics_plots_multiple_models(
        self, eval_metrics_results_per_model, model_names
    ):
        eval_metric_names = eval_metrics_results_per_model[0].keys()
        for metric_name in eval_metric_names:
            for split in eval_metrics_res[name].keys():
                save_name = metric_name +  '_combined_results_' + split + '.png'
                make_and_save_combined_metric_plots(
                    [
                        eval_metrics_res[metric_name][split] 
                        for eval_metrics_res in eval_metrics_results_per_model
                    ],
                    model_names,
                    save_name
                )

    def make_and_save_combined_metric_plots(self, metric_res_combined, model_names, save_name):
        metric_name = save_name.split('_')[0]
        start_times = metric_res_combined[0]['start_times']
        time_deltas = metric_res_combined[0]['time_deltas']
        values_combined = [metric_res['values'] for metric_res in metric_res_combined]
        eff_n = metric_res_combined[0]['eff_n']

        for s, start in enumerate(start_times):
            fig, axes = plt.subplots(2, 1, figsize=(5, 10))
            fig.suptitle(save_name + '_S=%d' %start)
            self.plot_eff_n_vs_deltas(
                axes[0], save_name, eff_n, 
                s, time_deltas
            )
            for v, values in enumerate(values_combined):
                self.plot_metric_values_vs_deltas(
                    axes[1], save_name, values, 
                    s, time_deltas, model_name=model_names[v]
                )
            save_path = os.path.join(self.savedir, save_name.split('.')[0] + '_S=%s.png' %start)
            plt.savefig(save_path)
