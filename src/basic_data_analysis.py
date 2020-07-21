import matplotlib.pyplot as plt
from utils.ParameterParser import ParameterParser
import torch
from data_handling.DataInput import DataInput
import numpy as np
import os

def make_data_analysis_plots(path_to_params, savedir):
    params = ParameterParser(path_to_params).parse_params()
    torch.random.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])
    torch.set_default_dtype(torch.float64)

    data_input = DataInput(params['data_input_params'])
    data_input.load_data()


    start_times = [0, .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]
    plot_cov_times_vs_event_times_for_diff_start_times(data_input, start_times, savedir)

    rayleigh_scale = 35.464
    plot_rayleigh_density_with_event_histogram(data_input, rayleigh_scale, savedir)

    start = 1.
    time_delta = .5
    frequency_vs_diagnosis_boxplot(data_input, start, time_delta, savedir)

def plot_cov_times_vs_event_times_for_diff_start_times(data_input, start_times, savedir):
    for s in start_times:
        plot_cov_times_vs_event_times_single_s(data_input, s, savedir)

def plot_cov_times_vs_event_times_single_s(data, start_time, savedir):

    bool_idxs_less_than_start = data.cov_times <= start_time
    idxs_most_recent_times = torch.max(torch.where(
     bool_idxs_less_than_start,
        data.cov_times, torch.zeros(data.cov_times.shape)
    ), dim=1)[1]
    most_recent_times = data.cov_times[torch.arange(idxs_most_recent_times.shape[0]), idxs_most_recent_times]

    sorted_idxs = torch.argsort(most_recent_times)
    sorted_most_recent_times = most_recent_times[sorted_idxs]
    sorted_event_times = data.event_times[sorted_idxs]

    plt.scatter(sorted_most_recent_times, sorted_event_times)
    plt.savefig(
        os.path.join(savedir, 'most_recent_cov_time_vs_event_times_at_S=%.3f.png' %start_time)
    )
    plt.clf()


def plot_rayleigh_density_with_event_histogram(data_input, rayleigh_scale, savedir):
    n_bins = 20
    plt.hist(
        data_input.event_times[~data_input.censoring_indicators.bool()], 
        n_bins, label='censored', density=True
    )
    plt.hist(
        data_input.event_times[data_input.censoring_indicators.bool()], n_bins, label='unecensored',
        alpha=.5, density=True
    )
    print(torch.sum(~data_input.censoring_indicators.bool()))
    print(data_input.event_times[~data_input.censoring_indicators.bool()].shape)
    start_x = 0
    end_x = 12

    x_axis_rayleigh = np.linspace(start_x, end_x, 50)
    pdf = \
        x_axis_rayleigh/rayleigh_scale * \
        np.exp(-x_axis_rayleigh**2 / (2 * rayleigh_scale))
    print(pdf)
    plt.plot(x_axis_rayleigh, pdf, label='rayleigh pdf')


    plt.legend()
    
    plt.savefig(
        os.path.join(savedir, 'rayleigh_with_event_time_hist.png')
    )
    plt.clf()

def frequency_vs_diagnosis_boxplot(data_input, start_time, time_delta, savedir):
    bool_idxs_less_than_start = data_input.cov_times <= start_time 

    idxs_most_recent_times = torch.max(torch.where(
        bool_idxs_less_than_start,
        data_input.cov_times, torch.zeros(data_input.cov_times.shape)
    ), dim=1)[1]
    last_times = data_input.cov_times[torch.arange(idxs_most_recent_times.shape[0]), idxs_most_recent_times]

    cov_times = data_input.cov_times
    num_visits = torch.sum(~(cov_times == 0), axis=1)
    freqs = last_times/(num_visits - 1)
         

    case_bool_idxs = \
     (start_time <= data_input.event_times) &\
     (data_input.event_times <= start_time + time_delta) &\
     (~data_input.censoring_indicators.bool())                    
    control_bool_idxs = data_input.event_times > start_time + time_delta

    case_freqs = freqs[case_bool_idxs].cpu().detach().numpy()
    control_freqs = freqs[control_bool_idxs].cpu().detach().numpy()

    case_freqs = case_freqs[~np.isnan(case_freqs)]
    control_freqs = control_freqs[~np.isnan(control_freqs)]
    plt.boxplot([case_freqs, control_freqs], labels=['case freq of visits', 'control freq of visits'], showfliers=False)

    plt.savefig(
        os.path.join(savedir, 'freq_vs_diagnosis_boxplot_S=%.3f_delta=%.3f.png' %(start_time, time_delta))
    )
    plt.clf()

if __name__ == '__main__':
    # only data loading params will be used so model is irrelevant
    path_to_params = '../configs/linear_delta_per_step.yaml'

    savedir = '../output/dm_cvd/data_analysis_plots/'
    make_data_analysis_plots(path_to_params, savedir)
