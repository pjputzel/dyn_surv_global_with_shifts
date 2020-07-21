import torch
import numpy as np
from utils.ParameterParser import ParameterParser
from main_types.BasicMain import BasicMain
from plotting.DynamicMetricsPlotter import DynamicMetricsPlotter

class MultiRunMain:

    def __init__(self, params):
        self.params = params

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        mains = []
        params_all = []
        for params_path in self.params['paths_to_configs']:
            params = ParameterParser(params_path).parse_params()
            main = BasicMain(params)
            main.main()
            mains.append(main)
            params_all.append(params)

        # now make combined plots
        evaluation_metrics = [main.results_tracker.eval_metrics for main in mains]
        plotter = DynamicMetricsPlotter(self.params['plot_params'], self.params['savedir'])
        model_names = [params['model_params']['model_type'] for params in params_all]
        plotter.make_and_save_dynamic_eval_metrics_plots_multiple_models(eval_metrics, model_names)

