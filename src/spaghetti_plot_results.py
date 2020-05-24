import pickle
import numpy
from plotting.ResultsPlotterSynth import ResultsPlotterSynth
import matplotlib.pyplot as plt

def plot_results(path_to_results):
    dist_type = 'exp'
    model_type = 'theta_per_step'
    true_params = [100, 10, .1]
    counts_per_group = [50, 50, 50]

    with open(path_to_results, 'rb') as f:
        diagnostics = pickle.load(f)

    pred_params_all_steps = diagnostics.pred_params_per_step
    if model_type == 'theta_per_step':
        last_param_preds = []
        for params_per_step in pred_params_all_steps:
            print(params_per_step.shape)
            print(params_per_step[1])
            padded_idx = params_per_step == 1.
            last_idx = 0
            for i, idx in enumerate(padded_idx):
                print(idx)
                if idx:
                    last_idx = i - 1
                    break
                    
            last_pred = params_per_step[last_idx]
            print(last_pred)
            last_param_preds.append(last_pred)
        pred_params = last_param_preds

    plot_params = {}
    results_plotter = ResultsPlotterSynth(pred_params, dist_type, plot_params)
    results_plotter.plot_learned_distribution_vs_true(true_params, counts_per_group)
    plt.savefig('new_learned_vs_true_pdf.png')
    plt.clf()
    results_plotter.make_learned_vs_true_boxplot(true_params, counts_per_group, labels=['rate'])
    plt.savefig('learned_vs_true_boxplot.png')

if __name__ == '__main__':
    plot_results('../output/synth/exp/tracker.pkl')
    
