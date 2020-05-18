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
    model_results = {}
    model_results['predicted_distribution_parameters'] = diagnostics.full_data_diagnostics['predicted_distribution_params']
    if model_type == 'theta_per_step':
        last_param_preds = []
        for params_per_step in model_results['predicted_distribution_parameters']:
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
        model_results['predicted_distribution_parameters'] = last_param_preds

    plot_params = {}
    results_plotter = ResultsPlotterSynth(model_results, dist_type, plot_params)
    print(len(diagnostics.full_data_diagnostics['predicted_distribution_params']))
    results_plotter.plot_learned_distribution_vs_true(true_params, counts_per_group)
    plt.savefig('new_learned_vs_true_pdf.png')
    plt.clf()
    results_plotter.make_learned_vs_true_boxplot(true_params, counts_per_group, labels=['rate'])
    plt.savefig('learned_vs_true_boxplot.png')

if __name__ == '__main__':
    plot_results('../output/synth/gamma/tracker.pkl')
    
