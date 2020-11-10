from main_types.BasicMain import BasicMain
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch

class DeltaPerStepResultsPlottingMain:
    
    def __init__(self, params):
        self.params = params
        self.setup_main = BasicMain(params)

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        self.setup_data_and_model()
        self.get_deltas()
        self.plot_deltas()
    
    def setup_data_and_model(self):
        data_input = self.setup_main.load_data()
        self.setup_main.preprocess_data(data_input)
        model = self.setup_main.load_model()

        self.data_input = data_input
        self.tr_data = data_input.get_tr_data_as_single_batch()
        self.te_data = data_input.get_te_data_as_single_batch()
        self.savedir = self.setup_main.params['savedir']
        model_path = os.path.join(self.savedir, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.model = model
        #print([param for param in model.parameters()])
        #print(torch.sum(model.linear.weight))

    def get_deltas(self):
        tr_deltas, _, _ = self.model(self.tr_data)
        self.tr_deltas = torch.squeeze(tr_deltas)
        te_deltas, _, _ = self.model(self.te_data)
        self.te_deltas = torch.squeeze(te_deltas)
    
    def plot_deltas(self, num_meas_times=10):
        timesteps = [i for i in range(10)]
        labels = ['meas%d' %i for i in timesteps]
        deltas_and_most_recent_idxs = [
            self.get_most_recent_deltas(self.te_deltas, self.te_data, time)
            for time in timesteps
        ]
        most_recent_deltas = [tpl[0] for tpl in deltas_and_most_recent_idxs]
        most_recent_idxs = [tpl[1] for tpl in deltas_and_most_recent_idxs]

        bool_idxs_to_keep_per_time = [
            np.where(
                most_recent_idxs[time] == time,
                np.ones(most_recent_idxs[time].shape[0]).astype(bool),
                np.zeros(most_recent_idxs[time].shape[0]).astype(bool)
            )
            for time in timesteps
        ]
        deltas_te = [
            most_recent_deltas[time][bool_idxs_to_keep_per_time[time]]
            for time in timesteps
        ]
        plt.boxplot(deltas_te, labels=labels)
        savepath = os.path.join(self.savedir, 'deltas_boxplots_te.png')
        plt.savefig(savepath)
        plt.clf()
    
        plt.boxplot(deltas_te[0])
        savepath = os.path.join(self.savedir, 'deltas_boxplot_te_t=0.png')
        plt.savefig(savepath)
    
        # TR plot
        deltas_and_most_recent_idxs = [
            self.get_most_recent_deltas(self.tr_deltas, self.tr_data, time)
            for time in timesteps
        ]
        most_recent_deltas = [tpl[0] for tpl in deltas_and_most_recent_idxs]
        most_recent_idxs = [tpl[1] for tpl in deltas_and_most_recent_idxs]

        bool_idxs_to_keep_per_time = [
            np.where(
                most_recent_idxs[time] == time,
                np.ones(most_recent_idxs[time].shape[0]).astype(bool),
                np.zeros(most_recent_idxs[time].shape[0]).astype(bool)
            )
            for time in timesteps
        ]
        deltas_tr = [
            most_recent_deltas[time][bool_idxs_to_keep_per_time[time]]
            for time in timesteps
        ]
        plt.boxplot(deltas_tr, labels=labels)
        savepath = os.path.join(self.savedir, 'deltas_boxplots_tr.png')
        plt.savefig(savepath)
        plt.clf()
    
        plt.boxplot(deltas_tr[0])
        savepath = os.path.join(self.savedir, 'deltas_boxplot_tr_t=0.png')
        plt.savefig(savepath)
        plt.clf()

        deltas_tr_one_individual = [d[0] for d in deltas_tr]
        plt.plot(np.arange(len(deltas_tr_one_individual)), deltas_tr_one_individual)
        savepath = os.path.join(self.savedir, 'ex_individual0_deltas_boxplots_tr.png')
        plt.savefig(savepath)
        plt.clf()

        deltas_tr_one_individual = [d[1] for d in deltas_tr]
        plt.plot(np.arange(len(deltas_tr_one_individual)), deltas_tr_one_individual)
        savepath = os.path.join(self.savedir, 'ex_individual1_deltas_boxplots_tr.png')
        plt.savefig(savepath)
        plt.clf()
    def get_most_recent_deltas(self, deltas, batch, start_time):
        max_times_less_than_start, idxs_max_times_less_than_start = \
            batch.get_most_recent_times_and_idxs_before_start(start_time)
 
        idxs_deltas = \
            [
                torch.arange(0, idxs_max_times_less_than_start.shape[0]),
                idxs_max_times_less_than_start
            ]
        deltas_at_most_recent_time = deltas[idxs_deltas].squeeze(-1)
        return deltas_at_most_recent_time.detach().numpy(), idxs_max_times_less_than_start.detach().numpy()
