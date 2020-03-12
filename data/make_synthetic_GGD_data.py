from scipy.stats import gengamma
from scipy.stats import expon
import pickle
import numpy as np
import matplotlib.pyplot as plt

# NOTE: It would be easy to generalize this to personalized analysis.
# For example we could model each covariate trajectory as the group trajectory
# plus a subject specific trajectory shift, and the locations as the group locations
# plus an individual shift.

def make_synthetic_data(num_individuals_per_group, 
                            subgroup_locs, 
                            subgroup_trajectory_funcs, 
                            poisson_rate,
                            censoring_prob,
                            scale=.0000001,
                            shape=100,
                            dist_type='gengamma'):
    num_groups = len(subgroup_locs)

    true_survival_times = get_true_survival_times(num_individuals_per_group, num_groups, subgroup_locs, scale, shape, dist_type=dist_type)
    censored_survival_times, censoring_indicators = randomly_censor_survival_times(true_survival_times, censoring_prob)

    trajectories = get_covariate_trajectories(censored_survival_times, subgroup_trajectory_funcs, poisson_rate)

    return flatten_trajectories_and_times(trajectories, censored_survival_times, censoring_indicators)


def get_true_survival_times(num_individuals_per_group, num_groups, subgroup_locs, scale, shape, dist_type='gengamma'):
    true_survival_times_per_group = [-1 * np.ones([int(num_individuals_per_group[i])]) for i in range(num_groups)]
    for group in range(num_groups):
        for individual in range(int(num_individuals_per_group[group])):
            if dist_type == 'gengamma':
                true_survival_times_per_group[group][individual] = gengamma.rvs(scale, shape, loc=subgroup_locs[group])
            elif dist_type == 'exp':
                true_survival_times_per_group[group][individual] = expon.rvs(scale=subgroup_locs[group])
            else:
                raise ValueError('Distribution type not recognized')
                
    return true_survival_times_per_group

def randomly_censor_survival_times(true_survival_times, censoring_prob, min_censoring_time_frac=.1):
    censoring_indicators = [-1 * np.ones([len(true_survival_times[i])]) for i in range(len(true_survival_times))]
    censored_survival_times = [-1 * np.ones([len(true_survival_times[i])]) for i in range(len(true_survival_times))]
    for group in range(len(true_survival_times)):
        for individual in range(true_survival_times[group].shape[0]):
            censored = True if np.random.ranf() > .5 else False
            censoring_indicators[group][individual] = censored
            if censored:
                censored_survival_times[group][individual] = np.random.uniform(min_censoring_time_frac * true_survival_times[group][individual], true_survival_times[group][individual])
            else:
                censored_survival_times[group][individual] = true_survival_times[group][individual]
    return censored_survival_times, censoring_indicators

def get_covariate_trajectories(censored_survival_times, subgroup_trajectory_funcs, poisson_rate):
    trajectories = []
    for group in range(len(censored_survival_times)):
        trajectories.append([])
        for individual in range(censored_survival_times[group].shape[0]):
            trajectory = sample_single_trajectory(poisson_rate, subgroup_trajectory_funcs[group], censored_survival_times[group][individual])
            trajectories[group].append(trajectory)
    return trajectories

def sample_single_trajectory(poisson_rate, trajectory_path_func, censored_survival_time, noise_mean=0., noise_scale=.00001, min_len=5, max_len=100, max_attempts=10):
    events_len = 0
    print(censored_survival_time)
    num_attempts = 0
    temp_rate = poisson_rate
    while events_len < min_len or events_len > max_len:
        if num_attempts > max_attempts:
            print(events_len, temp_rate, censored_survival_time)
            if events_len < min_len:
                print('too few events, increasing rate')
                temp_rate = temp_rate * .5
            if events_len > max_len:
                print('too many events, decreasing rate')
                temp_rate = temp_rate * 2
        next_time = np.random.exponential(scale=poisson_rate)
        # make sure the first time is bigger than the survival time
        while next_time > censored_survival_time:
            next_time = np.random.exponential(scale=temp_rate)
        events = []
        while next_time <= censored_survival_time:
            events.append([next_time])
            events[-1].append(trajectory_path_func(next_time) + np.random.normal(noise_mean, noise_scale))
            next_time = next_time + np.random.exponential(scale=temp_rate)
            if len(events) > max_len:
                break
        events_len = len(events)
        num_attempts += 1
    return events


def flatten_trajectories_and_times(trajectories_per_group, censored_survival_times, censoring_indicators):
    trajectories_per_individual = trajectories_per_group[0]
    censored_survival_times_per_individual = censored_survival_times[0]
    censoring_indicators_per_individual = censoring_indicators[0]
    for i in range(1, len(trajectories_per_group)):
        trajectories_per_individual.extend(trajectories_per_group[i])
        censored_survival_times_per_individual = np.concatenate(\
            [censored_survival_times_per_individual, censored_survival_times[i]])
        censoring_indicators_per_individual = np.concatenate(\
            [censoring_indicators_per_individual, censoring_indicators[i]])

    return trajectories_per_individual, censored_survival_times_per_individual, censoring_indicators_per_individual

def get_linear_trajectory_func(slope, intercept):

    def linear_trajectory_func(x):
        return slope * x + intercept

    return linear_trajectory_func

def get_sine_trajectory_func(period):
    def sine_trajectory_func(x):
        return 5 * np.sin(x * 2  * np.pi/period)
    return sine_trajectory_func

if __name__ == '__main__':
    num_groups = 3
    num_individuals_per_group = 100 * np.ones(num_groups)
    subgroup_locs = [10, 20, 30] # corresponding to rates 10, 5, and 1 for exponential
    subgroup_trajectory_funcs_linear = [get_linear_trajectory_func(1, 1), get_linear_trajectory_func(1, 3), get_linear_trajectory_func(1, 5)]
    subgroup_trajectory_funcs_sine = [get_sine_trajectory_func(4 * np.pi) , get_sine_trajectory_func(2 * np.pi), get_sine_trajectory_func(8 * np.pi)]

    #num_groups = 1
    #num_individuals_per_group = [50.]
    #subgroup_locs = [5.]

    poisson_scale = .1
    censoring_prob = 0
    #subgroup_trajectory_funcs_linear = [get_linear_trajectory_func(1, 0)]
    
    trajectories, censored_survival_times, censoring_indicators = make_synthetic_data(num_individuals_per_group, subgroup_locs, subgroup_trajectory_funcs_linear, poisson_scale, censoring_prob, dist_type='gamma')

    traj_len_counts = np.zeros(np.max([len(traj) for traj in trajectories]))
    traj_lens = []
    for traj in trajectories:
        traj_len_counts[len(traj) - 1] += 1
        traj_lens.append(len(traj))
#    print([(key, traj_lens[key]) for key in np.sort([key for key, value in traj_lens.items()])])
#    print(np.sum([value for key, value in traj_lens.items()]))
    plt.hist(traj_lens, bins=np.max([len(traj) for traj in trajectories]))
    plt.savefig('histogram_synth_traj_lens.png')
    plt.clf()
    plt.hist(censored_survival_times, bins=100)
    plt.savefig('histogram_censored_survival_times.png')
    plt.clf()
    with open('./synth/trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)

    with open('./synth/censored_survival_times.pkl', 'wb') as f:
        pickle.dump(censored_survival_times, f)

    with open('./synth/censoring_indicators.pkl', 'wb') as f:
        pickle.dump(censoring_indicators, f)


    print(trajectories[0], censored_survival_times[0], censoring_indicators[0])
    print(trajectories[-1], censored_survival_times[-1], censoring_indicators[-1])
    for t, traj in enumerate(trajectories):
        if t <= 50:
            plt.plot([traj[i][0] for i in range(len(traj))], [traj[i][1] for i in range(len(traj))], c='r')
        elif t <= 100:
            plt.plot([traj[i][0] for i in range(len(traj))], [traj[i][1] for i in range(len(traj))], c='b')
        else:
            plt.plot([traj[i][0] for i in range(len(traj))], [traj[i][1] for i in range(len(traj))], c='g')
            
            
    plt.savefig('Synthetic_Covariate_Trajectories.png')
    plt.clf()

    g1_rvs = expon.rvs(scale=subgroup_locs[0], size=1000)
    g2_rvs = expon.rvs(scale=subgroup_locs[1], size=1000)
    g3_rvs = expon.rvs(scale=subgroup_locs[2], size=1000)

    fig, axes = plt.subplots(3,  sharey=True)
    axes[0].hist(g1_rvs, bins=50, density=True)
    #plt.savefig('g1_rvs.png')
    #plt.clf()
    
    axes[1].hist(g2_rvs, bins=50, density=True)
    #plt.savefig('g2_rvs.png')
    #plt.clf()

    axes[2].hist(g3_rvs, bins=50, density=True)
    #plt.savefig('g3_rvs.png')
    #plt.clf()
    plt.savefig('all_three_rvs.png')
