import pickle
import os
import torch
import numpy
from plotting.ResultsPlotterSynth import ResultsPlotterSynth
import matplotlib.pyplot as plt

def plot_results(path_to_results):
    dist_type = 'exponential'
    model_type = 'theta_per_step'
    true_params = [[10], [1], [.1]]
    counts_per_group = [50, 50, 50]

    with open(os.path.join(path_to_results, 'tracker.pkl'), 'rb') as f:
        diagnostics = pickle.load(f)

    pred_params_last_train_step = diagnostics.pred_params_per_step[-1]
    pred_params = torch.mean(pred_params_last_train_step, dim=1)
    print(pred_params.shape)

    plot_params = {}
    results_plotter = ResultsPlotterSynth(pred_params, dist_type, plot_params)
    results_plotter.plot_learned_distribution_vs_true(true_params, counts_per_group)
    plt.savefig(os.path.join(path_to_results, 'learned_vs_true_pdf.png'))
    plt.clf()
    results_plotter.make_learned_vs_true_boxplot(true_params, counts_per_group, labels=['rate'])
    plt.savefig(os.path.join(path_to_results, 'learned_vs_true_boxplot.png'))

if __name__ == '__main__':
    plot_results('../output/synth/exp/')
    
