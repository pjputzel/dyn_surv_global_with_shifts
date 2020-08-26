import matplotlib.pyplot as plt
from utils.ParameterParser import ParameterParser
import torch
from data_handling.DataInput import DataInput
import numpy as np
import os
from pandas import qcut

def make_data_analysis_plots(path_to_params, savedir):
    params = ParameterParser(path_to_params).parse_params()
    torch.random.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])
    torch.set_default_dtype(torch.float64)

    data_input = DataInput(params['data_input_params'])
    data_input.load_data()

    start_times = [.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]
    num_events_buckets = [1,2,3,4,5,6]
    plot_event_times_stratified_by_num_events_start_times(\
        data_input, start_times, num_events_buckets,
        savedir
    )

    plot_cov_times_vs_event_times_for_diff_start_times(data_input, start_times, savedir)

    rayleigh_scale = 26.949
    plot_rayleigh_density_with_event_histogram(data_input, rayleigh_scale, savedir)

    start = .5
    time_delta = 1.5
    frequency_vs_diagnosis_boxplot(data_input, start, time_delta, savedir)

    start = 1.
    time_delta = 1.5
    frequency_vs_diagnosis_boxplot(data_input, start, time_delta, savedir)

    start = 2.
    time_delta = 1.5
    frequency_vs_diagnosis_boxplot(data_input, start, time_delta, savedir)

    
    start = 2.
    time_delta = 3.0
    frequency_vs_diagnosis_boxplot(data_input, start, time_delta, savedir)



def plot_event_times_stratified_by_num_events_start_times(
    data_input, start_times, num_events_buckets,
    savedir, plot_uncensored_after_S_only=True
    ):

    for start in start_times:
        plot_event_times_stratified_by_num_events(
            data_input, start, num_events_buckets,
            savedir, plot_uncensored_after_S_only=plot_uncensored_after_S_only
        )
def plot_event_times_stratified_by_num_events(
    data_input, start_time, num_events_buckets, 
    savedir, plot_uncensored_after_S_only=True
    ):
    _, idxs_last_times = data_input.get_most_recent_times_and_idxs_before_start(start_time) 
    num_events = idxs_last_times + 1 # + 1 is accounting for zero indexing

    valid_bool_idxs = torch.ones(data_input.event_times.shape, dtype=torch.bool)
    if plot_uncensored_after_S_only:
        valid_bool_idxs = \
            (data_input.event_times > start_time) &\
            (~data_input.censoring_indicators.bool())
    num_events = num_events[valid_bool_idxs]
    event_times = data_input.event_times[valid_bool_idxs] - start_time

    labels = []
    bucketed_num_events = []
    bucketed_event_times = []    
    for bucket in num_events_buckets:
        bucket_idxs = num_events == bucket
        num_events_bucket = num_events[bucket_idxs]
        event_times_bucket = event_times[bucket_idxs]

        bucketed_num_events.append(num_events_bucket)
        bucketed_event_times.append(event_times_bucket)
        labels.append('J=%d, N=%d   ' %(bucket, torch.sum(bucket_idxs)))

    # for last bucket take everything greater than last num events level
    last_bucket_idxs = num_events > num_events_buckets[-1]
    bucketed_num_events.append(num_events[last_bucket_idxs])
    bucketed_event_times.append(event_times[last_bucket_idxs])
    labels.append('J > %d, N=%d' %(bucket, torch.sum(last_bucket_idxs)))

    bucketed_event_times = [b.detach().cpu().numpy() for b in bucketed_event_times]
    plt.boxplot(\
        bucketed_event_times, labels=labels, 
        positions=range(1, 3 * len(bucketed_event_times) + 1, 3)
    )
    plt.ylabel('True Event Times (relative to S)')
    plt.xlabel('Number of Covariate Measurements (J) Before S=%.3f' %(start_time))
    if plot_uncensored_after_S_only:
        plt.title('Uncensored Data Only-Visits vs True Event Times')
    else:
        plt.title('All Data (censored and uncensored) Visits vs True Event Times')
        


#    plt.scatter(sorted_most_recent_times, sorted_event_times)
    #plt.hist(sorted_most_recent_times, sorted_event_times, bins=20)
    plt.savefig(
        os.path.join(savedir, 'event_times_stratified_by_num_cov_events_at_S=%.3f.png' %start_time)
    )
    plt.clf()

    


def plot_cov_times_vs_event_times_for_diff_start_times(data_input, start_times, savedir, num_bins=4):
    for s in start_times:
        plot_cov_times_vs_event_times_single_s(data_input, s, savedir, num_bins=num_bins)

def plot_cov_times_vs_event_times_single_s(data, start_time, savedir, num_bins=3):

    _, idxs_last_times = data.get_most_recent_times_and_idxs_before_start(start_time)
    # For conditioning on \tau_i > S
    idxs_last_times = idxs_last_times[
        (~data.censoring_indicators.bool()) &\
        (data.event_times > start_time)
    ]

    num_events = idxs_last_times + 1
    sorted_idxs = torch.argsort(num_events)
    sorted_num_events = num_events[sorted_idxs]
    sorted_event_times = data.event_times[
        (~data.censoring_indicators.bool()) &\
        (data.event_times > start_time)
    ][sorted_idxs]
    # make the event times relative to the starting time
    sorted_event_times = (sorted_event_times - start_time) * 365.

    max_num_events = torch.max(num_events)
    # for using numpy to split bins
#    bins = np.linspace(0, max_num_events, num_bins + 1)

    # for using pandas qcut
    bins_per_person = qcut(\
     sorted_num_events.cpu().detach().numpy(), num_bins,
#     duplicates='drop'
    )
    #print(bins_per_person.value_counts())
    bins_per_person = bins_per_person.to_list()
    bin_ends_per_person = [b.right for b in bins_per_person]
    bin_ends = np.unique(bin_ends_per_person)
    bin_labels = ['%.2f to %.2f' %(bin_ends[i] + 1, bin_ends[i + 1]) for i in range(num_bins - 1)]
    bin_labels.insert(0, '%.2f to %.2f' %(1, bin_ends[0]))
#    print(bin_ends)
#    print(np.where(bin_ends == bin_ends[0])[0][0])
    bin_memberships = np.array([np.where(bin_ends == bin_i)[0][0] for bin_i in bin_ends_per_person])
    print(bin_memberships)
#    bin_labels.append(str(bins[i]) + 'to%.2f' %start_time)
#    bin_memberships = np.digitize(sorted_num_events, bins=bins[0:num_bins])
    event_times_all = []
    num_events_all = []
    num_per_group = []
    for i in range(num_bins):
        group_i_bool = bin_memberships == i # + 1 for np version
        events_times_i = sorted_event_times[group_i_bool]
        num_events_i = sorted_num_events[group_i_bool]
        
        event_times_all.append(events_times_i.detach().cpu().numpy())
        num_events_all.append(num_events_i.detach().cpu().numpy())
        num_per_group.append(np.sum(group_i_bool))
    print(num_per_group)
    #labels = [l + ', N=%d' %num_per_group[i] for i, l in enumerate(bin_labels)]
    plt.boxplot(event_times_all, labels=bin_labels)
    plt.ylabel('True Event Times in Days (relative to S)')
    plt.xlabel('Number of Covariate Measurements Before S=%.3f, N-Per-Quantile: %d' %(start_time, num_per_group[0]))
    plt.title('Uncensored Data Only-Visits vs True Event Times')    
        


#    plt.scatter(sorted_most_recent_times, sorted_event_times)
    #plt.hist(sorted_most_recent_times, sorted_event_times, bins=20)
    plt.savefig(
        os.path.join(savedir, 'most_recent_cov_time_vs_event_times_at_S=%.3f.png' %start_time)
    )
    plt.clf()


def plot_rayleigh_density_with_event_histogram(data_input, rayleigh_scale, savedir):
    n_bins = 20
    plt.hist(
        data_input.event_times[~data_input.censoring_indicators.bool()], 
        n_bins, label='uncensored', density=True
    )
    plt.hist(
        data_input.event_times[data_input.censoring_indicators.bool()], n_bins, label='censored',
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
    last_times, idxs_last_times = data_input.get_most_recent_times_and_idxs_before_start(start_time)


    cov_times = data_input.cov_times
    avg_freqs = []
    nan_idxs = []
    for i, idx_last_time in enumerate(idxs_last_times):
        avg_freq_i = 0
        for j in range(idx_last_time):
            period = cov_times[i, j + 1] - cov_times[i, j]
            freq = 1./period
            avg_freq_i += freq
        avg_freqs.append(avg_freq_i/idx_last_time)
        if idx_last_time == 0:
            nan_idxs.append(1)
        else:
            nan_idxs.append(0)
    freqs = torch.tensor(avg_freqs)
    nan_idxs = torch.tensor(nan_idxs).bool()
#    num_visits = torch.sum(~(cov_times == 0), axis=1)
#    freqs = last_times/(num_visits - 1)
         

    case_bool_idxs = \
     (start_time <= data_input.event_times) &\
     (data_input.event_times <= start_time + time_delta) &\
     (~data_input.censoring_indicators.bool())                    
    control_bool_idxs = \
    (data_input.event_times > start_time + time_delta) &\
    (~data_input.censoring_indicators.bool())
    

    
    case_num_events = idxs_last_times[case_bool_idxs].cpu().detach().numpy() + 1
    control_num_events = idxs_last_times[control_bool_idxs].cpu().detach().numpy() + 1

#    case_freqs = freqs[case_bool_idxs & ~nan_idxs].cpu().detach().numpy()
#    control_freqs = freqs[control_bool_idxs & ~nan_idxs].cpu().detach().numpy()

#    case_freqs = case_freqs[~np.isnan(case_freqs)]
#    control_freqs = control_freqs[~np.isnan(control_freqs)]
#    plt.boxplot([case_freqs, control_freqs], labels=['Cases', 'Controls'], showfliers=False)
    plt.boxplot([case_num_events, control_num_events], labels=['Cases', 'Controls'], showfliers=False)
    plt.ylabel('Num Visits Before Year S')

    plt.savefig(
        os.path.join(savedir, 'num_visits_vs_diagnosis_boxplot_S=%.3f_delta=%.3f.png' %(start_time, time_delta))
    )
    plt.clf()

if __name__ == '__main__':
    # only data loading params will be used so model is irrelevant
    path_to_params = '../configs/linear_baseline_configs/linear_delta_per_step.yaml'

    savedir = '../output/dm_cvd/data_analysis_plots/'
    make_data_analysis_plots(path_to_params, savedir)
