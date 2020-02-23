import pickle
import numpy
from plotting.ResultsPlotterSynth import ResultsPlotterSynth
import matplotlib.pyplot as plt

def plot_results_exp(path_to_results):
    with open(path_to_results, 'rb') as f:
        diagnostics = pickle.load(f)
    model_results = {}
    model_results['predicted_distribution_parameters'] = diagnostics.cur_diagnostics['predicted_distribution_params']
    model_type = 'exp'
    plot_params = {}
    results_plotter = ResultsPlotterSynth(model_results, model_type, plot_params)
    results_plotter.plot_learned_distribution_vs_true([1/.1, 1/.5, 1.], [50, 50, 50])
    plt.savefig('learned_vs_true_pdf.png')

if __name__ == '__main__':
    plot_results_exp('../output/synth/tracker.pkl')
    
