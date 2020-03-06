import pickle
import numpy
from plotting.ResultsPlotterSynth import ResultsPlotterSynth
import matplotlib.pyplot as plt

def plot_results_exp(path_to_results):
    with open(path_to_results, 'rb') as f:
        diagnostics = pickle.load(f)
    model_results = {}
    model_results['predicted_distribution_parameters'] = diagnostics.full_data_diagnostics['predicted_distribution_params']
    model_type = 'gamma'
    plot_params = {}
    results_plotter = ResultsPlotterSynth(model_results, model_type, plot_params)
    gamma_true_params = [[100, 10], [100, 20], [100, 30]]
    results_plotter.plot_learned_distribution_vs_true(gamma_true_params, [2, 2, 2])
    plt.savefig('learned_vs_true_pdf.png')

if __name__ == '__main__':
    plot_results_exp('../output/synth/tracker.pkl')
    
