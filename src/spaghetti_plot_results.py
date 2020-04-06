import pickle
import numpy
from plotting.ResultsPlotterSynth import ResultsPlotterSynth
import matplotlib.pyplot as plt

def plot_results_exp(path_to_results):
    with open(path_to_results, 'rb') as f:
        diagnostics = pickle.load(f)
    model_results = {}
    model_results['predicted_distribution_parameters'] = diagnostics.full_data_diagnostics['predicted_distribution_params']
    model_type = 'ggd'
    plot_params = {}
    results_plotter = ResultsPlotterSynth(model_results, model_type, plot_params)
    gamma_true_params = [[float(1e-7), 100,  10], [float(1e-7), 100, 20], [float(1e-7), 100, 30]]
    counts_per_group = [200, 200, 200]
    print(len(diagnostics.full_data_diagnostics['predicted_distribution_params']))
    results_plotter.plot_learned_distribution_vs_true(gamma_true_params, counts_per_group)
    plt.savefig('learned_vs_true_pdf.png')
    plt.clf()
    results_plotter.make_learned_vs_true_boxplot(gamma_true_params, counts_per_group, labels=['alpha', 'beta'])
    plt.savefig('learned_vs_true_boxplot.png')

if __name__ == '__main__':
    plot_results_exp('../output/synth/gamma/tracker.pkl')
    
