import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


def main():
    n_per_class = [10000, 10000]
    # for theta per step
#    dataset_type = 'theta_per_step'
#    theta_func = theta_max_one_half_linear
#    cov_traj_func_per_class = \
#        [
#            linear_func_slope2,
#            linear_func_slope_one_half
#        ]
#    interevent_rate_param_per_class = [.1, .2]
#    global_theta = None

    # for delta per step
    dataset_type = 'delta_per_step'
    delta_func = linear_func_slope1
    #maybe healthy should age half as slow (ie 1/2t)
    cov_traj_func_per_class = \
        [
            linear_func_slope20,
            linear_func_slope1
        ]
    interevent_rate_param_per_class = [.05, .1]
    global_theta = .1

    data = SimpleData(
        n_per_class, delta_func, cov_traj_func_per_class,
        interevent_rate_param_per_class, dataset_type=dataset_type,
        global_theta=global_theta
    )
    data.construct_data()
    with open('synth/simple_synth_%s.pkl' %dataset_type, 'wb') as f:
        pickle.dump(data, f)


    print(data.cov_trajs[0], data.true_event_times[0], data.thetas[0])
    print(data.cov_trajs[-1], data.true_event_times[-1], data.thetas[-1])
    plt.hist(
        data.true_event_times[0:n_per_class[0]], alpha=.5, 
        label='unhealthy', bins=50, density=True
    )
    plt.hist(
        data.true_event_times[n_per_class[0]:], alpha=.5, 
        label='healthy', bins=50, density=True
    )
    plt.legend()
    plt.savefig('hist_by_class.png')
    plt.clf()
    plt.plot(data.cov_times[0], data.cov_values[0], label='unhealthy ex cov traj')
    plt.plot(data.cov_times[n_per_class[0]], data.cov_values[n_per_class[0]], label='healthy ex cov traj')
    plt.legend()
    plt.savefig('cov_trajs_ex.png')
    plt.clf()
#    x = np.linspace(0, 1, 100)
#    plt.plot(x, (x/global_theta) * np.exp(-x**2/(2 * global_theta)))
#    plt.savefig('global_density.png')

def test_truncated_sampling():
    n_per_class = [1000, 1000]
    theta_func = theta_max_one_half_linear
    cov_traj_func_per_class = \
        [
            linear_func_slope5,
            linear_func_slope_one_half
        ]
    interevent_rate_param_per_class = [.1, .2]
    dataset_type = 'delta_per_step'
    global_theta = 1.
    data = SimpleData(
        n_per_class, theta_func, cov_traj_func_per_class,
        interevent_rate_param_per_class, dataset_type=dataset_type,
        global_theta=global_theta
    )
    samples = []
    trunc_time = .2
    shift = 2.
    for i in range(100000):
        samples.append(data.sample_from_truncated_rayleigh(shift, trunc_time))
    plt.hist(samples, label='truncation at time %.1f with %.1f shift' %(trunc_time, shift), density=True, bins=100)
    x = np.linspace(0, 10, 100)
    plt.plot(x, (x/global_theta) * np.exp(-x**2/(2 * global_theta)), label='complete density')
    plt.legend()
    plt.savefig('truncation_sampling_test.png')

def zero_func(x):
    return 0.

def quadratic_func(x):
    return x ** 2

def square_root_func(x):
    return x ** (1/2)

def quartic_func(x):
    return x ** 4

def delta_func_one_half_slope(x):
    return x/2

def delta_func_one_slope_two(x):
    return x * 2

def theta_max_one_half_linear(x):
    if x >= .5:
        return 0.01
    return .5 - x

def base_two_exponential(x):
    return 2 ** x

def theta_max5_linear(x):
    if x >= 5:
        return 0.001
    return 5 - x

def inverse_func(x):
    if x == 0:
        return 0
    return 1/x

def linear_func_slope1(x):
    return x

def linear_func_slope2(x):
    return x * 2

def linear_func_slope_one_half(x):
    return .5 * x

def linear_func_slope_one_eighth(x):
    return (1/8) * x
def linear_func_slope5(x):
    return 5 * x

def linear_func_slope20(x):
    return 20 * x
def linear_func_slope_one_one_hundreth(x):
    return (1/100) * x

def linear_func_slope_one_ten_thousandth(x):
    return (1/10000) * x

def linear_func_slope_one_trillionth(x):
    return (1./1000000000) * x

class SimpleData:

    def __init__(self,
        n_per_class, theta_func, cov_traj_func_per_class,
        interevent_rate_param_per_class, 
        censoring_prob_per_class=None, dist_type='rayleigh',
        global_theta=None, dataset_type='theta_per_step'
    ):
        self.n_classes = len(n_per_class)
        self.n_per_class = n_per_class
        self.theta_func = theta_func
        self.cov_traj_func_per_class = cov_traj_func_per_class
        self.interevent_rate_param_per_class = interevent_rate_param_per_class
        self.censoring_prob_per_class = censoring_prob_per_class
        self.dist_type = dist_type
        self.dataset_type = dataset_type
        if self.dataset_type == 'delta_per_step' and (global_theta is None):
            raise ValueError('Delta per step dataset type requires specification of keyword global theta')
        self.global_theta = global_theta

    def construct_data(self):
        #self.clear_data()
        self.sample_trajs_and_event_times()
        # must censor after sampling cov measurements
        # to prevent correlation between the censoring times
        # and the number of events per person
        self.randomly_censor_event_times()
        self.truncate_cov_seqs()
        self.form_cov_trajectories()
    

    def sample_trajs_and_event_times(self):
        cov_vals, cov_times, true_event_times, thetas = [], [], [], []
        for c, n in enumerate(self.n_per_class):
            for i in range(n):
                cov_vals_i, cov_times_i, thetas_i, true_event_time_i = \
                    self.sample_single_traj_and_event_time(c)
                cov_vals.append(cov_vals_i)
                cov_times.append(cov_times_i)
                thetas.append(thetas_i)
                true_event_times.append(true_event_time_i)
        self.cov_values = cov_vals
        self.cov_times = cov_times
        self.thetas = thetas
        self.true_event_times = true_event_times

    def sample_single_traj_and_event_time(self, class_idx):
        cov_vals = [self.cov_traj_func_per_class[class_idx](0)]
        cov_times = [0]
        thetas = [self.theta_func(cov_vals[0])]
        
        proposed_time = np.inf 
        time_delta = self.sample_time_delta_per_class(class_idx)
        while cov_times[-1] + time_delta < proposed_time:
            next_cov_time = cov_times[-1] + time_delta
            next_cov_val = self.cov_traj_func_per_class[class_idx](next_cov_time)
            next_theta = self.theta_func(next_cov_val)
            
            cov_times.append(next_cov_time)
            cov_vals.append(next_cov_val)
            thetas.append(next_theta)         

            proposed_time = self.sample_proposed_time(
                next_theta, next_cov_time
            )
            time_delta = self.sample_time_delta_per_class(class_idx)
        true_event_time = proposed_time
        return cov_vals, cov_times, thetas, true_event_time
    
    def sample_time_delta_per_class(self, class_idx):
        rate = self.interevent_rate_param_per_class[class_idx]
        return np.random.exponential(rate)

    def sample_proposed_time(self, theta, next_cov_time):
        # this should be the only real difference between the
        # two datasets!

        if self.dataset_type == 'theta_per_step':
            proposed_time = np.random.__getattribute__(
                self.dist_type
            )(*[theta**(1/2)])
            proposed_time += next_cov_time
        elif self.dataset_type == 'delta_per_step':
            proposed_time = self.sample_truncated_distribution(
                theta, next_cov_time
            )
        else:
            raise ValueError('Dataset type %s not recognized' %self.dataset_type)
        return proposed_time

    def sample_truncated_distribution(self, delta, truncation_time):
        if self.dist_type == 'rayleigh':
            proposed_time = \
                self.sample_from_truncated_rayleigh(delta, truncation_time)
        else:
            raise ValueError('Distribution type %s not recognized' %self.dist_type)
        return proposed_time

    def sample_from_truncated_rayleigh(self, delta, truncation_time):
        CDF = np.random.uniform()
        discriminant = \
            4 * delta**2 +\
            4 * (2 * delta * truncation_time + truncation_time**2 + \
                2*self.global_theta * np.log(1/(1 - CDF)))
        sample = (-2 * delta + discriminant**(1/2))/2
        return sample

#    def sample_true_event_times(self):
#        event_times = []
#        classes_per_person = []
#        for c, n in enumerate(self.n_per_class):
#            classes_per_person.extend([c for i in range(n)])
#            event_times.extend(self.sample_event_times_for_class_c(c, n))
#        self.true_event_times = event_times
#        self.classes_per_person = classes_per_person
#
#    def sample_event_times_for_class_c(self, class_idx, n):
#        event_times = np.random.__getattribute__(
#            self.dist_type
#        )(*self.params_per_class[class_idx], n)
#        return list(event_times)
#
#    def sample_cov_measurement_times(self):
#        cov_times= []
#        for i, time in enumerate(self.true_event_times):
#            cur_time = 0
#            c = self.get_class(i)
#            cov_times_i = []
#            while cur_time < self.true_event_times[i]:
#                cov_times_i.append(cur_time)
#                delta = np.random.exponential(
#                    self.interevent_rate_param_per_class[c]
#                )
#                cur_time += delta
#            cov_times.append(cov_times_i)
#        self.cov_times = cov_times

#    def compute_cov_measurement_values(self):
#        cov_values = []
#        for i in range(len(self.true_event_times)):
#            cov_values_i = []
#            for cov_measurement_time in self.cov_times[i]:
#                c = self.get_class(i)
#                cov_values_i.append(
#                    self.cov_traj_func_per_class[c](cov_measurement_time)
#                )
#            cov_values.append(cov_values_i)
#        self.cov_values = cov_values
                
    def form_cov_trajectories(self):
        cov_trajs = []
        for i in range(len(self.cov_times)):
            cov_trajs_i = []
            for time, val in zip(self.cov_times[i], self.cov_values[i]):
                cov_trajs_i.append([time, val]) 
            cov_trajs.append(cov_trajs_i)
        self.cov_trajs = cov_trajs

#    def sample_number_of_cov_events_per_person(self):
#        num_cov_events[]
#        for i, time in enumerate(self.true_event_times):
#            c = self.get_class(i)
#            n_events_i = np.random.poisson(
#                self.interevent_rate_param_per_class[c] * \
#                self.true_event_times[i]
#            )
#            num_cov_events.append(n_events_i)
#        self.n_cov_events = num_cov_events
    
   
    def get_class(self, i):
        return self.classes_per_person[i]
         
    def randomly_censor_event_times(self):
        # for now not using censored data
        pass
    
    def truncate_cov_seqs(self):
        # no censoring currently implemented
        # so no truncation needed
        pass


class LearnedDataThetaIJ(SimpleData):
    def __init__(self, trained_model, simple_data):
        self.n_classes = len(simple_data.n_per_class)
        self.n_per_class =  simple_data.n_per_class
        self.model = trained_model
        self.theta_func = self.get_theta_func()
        if not simple_data.dataset_type == 'theta_per_step':
            raise ValueError('LearnedDataThetaIJ must be initialized with a delta_per_step dataset!')
        self.dataset_type = simple_data.dataset_type
        self.cov_traj_func_per_class = simple_data.cov_traj_func_per_class
        self.interevent_rate_param_per_class = simple_data.interevent_rate_param_per_class
        self.censoring_prob_per_class = simple_data.censoring_prob_per_class
        self.dist_type = simple_data.dist_type
        self.global_theta = None
    
    def get_theta_func(self):
        model_class_name = type(self.model).__name__
        if model_class_name == 'LinearThetaIJModel':
            theta_func = self.get_theta_func_linear()
        else:
            error = '%s model type not yet implemented for LearnedDataClass' %model_class_name
            raise NotImplementedError(error)
        return theta_func

    def get_theta_func_linear(self):
        coef_time = self.model.linear.weight[0][0]
        coef_cov = self.model.linear.weight[0][1]
        bias = self.model.linear.bias
        def theta_func(cov, t):
            def softplus(x, B=100):
                return torch.nn.functional.softplus(x, B)
            activation = coef_time * t + coef_cov * cov + bias
#            ret = torch.exp(-activation).cpu().detach().numpy()[0]
            ret = softplus(activation).cpu().detach().numpy()[0]
            return ret
        return theta_func

    def sample_single_traj_and_event_time(self, class_idx, max_traj_len=1000):
        cov_vals = [self.cov_traj_func_per_class[class_idx](0)]
        cov_times = [0]
        thetas = [self.theta_func(cov_vals[0], 0)]
        
        proposed_time = np.inf 
        time_delta = self.sample_time_delta_per_class(class_idx)
        i = 0
        continue_loop = (cov_times[-1] + time_delta < proposed_time) and (i < max_traj_len)
        while continue_loop:
            next_cov_time = cov_times[-1] + time_delta
            next_cov_val = self.cov_traj_func_per_class[class_idx](next_cov_time)
            # main difference from SimpleData class
            next_theta = self.theta_func(next_cov_val, next_cov_time)
            
            cov_times.append(next_cov_time)
            cov_vals.append(next_cov_val)
            thetas.append(next_theta)         

            proposed_time = self.sample_proposed_time(
                next_theta, next_cov_time
            )
            time_delta = self.sample_time_delta_per_class(class_idx)
            i += 1
            continue_loop = (cov_times[-1] + time_delta < proposed_time) and (i < max_traj_len)

        if i >= max_traj_len:
            print('Max trajectory length reached while sampling from learned data!')
        true_event_time = proposed_time
        return cov_vals, cov_times, thetas, true_event_time

class LearnedDataDeltaIJ(SimpleData):
    
    def __init__(self, trained_model, simple_data):
        self.n_classes = len(simple_data.n_per_class)
        self.n_per_class =  simple_data.n_per_class
        self.model = trained_model
        self.theta_func = self.get_theta_func()
        if not simple_data.dataset_type == 'delta_per_step':
            raise ValueError('LearnedDataDeltaIJ must be initialized with a delta_per_step dataset!')
        self.dataset_type = simple_data.dataset_type
        self.cov_traj_func_per_class = simple_data.cov_traj_func_per_class
        self.interevent_rate_param_per_class = simple_data.interevent_rate_param_per_class
        self.censoring_prob_per_class = simple_data.censoring_prob_per_class
        self.dist_type = simple_data.dist_type
        self.global_theta = torch.exp(-self.model.global_param_logspace).cpu().detach().numpy()[0]

    def get_theta_func(self):
        model_class_name = type(self.model).__name__
        if model_class_name == 'LinearDeltaIJModel':
            theta_func = self.get_theta_func_linear()
        else:
            error = '%s model type not yet implemented for LearnedDataClass' %model_class_name
            raise NotImplementedError(error)
        return theta_func
        
    def get_theta_func_linear(self):
        coef_time = self.model.linear.weight[0][0]
        coef_cov = self.model.linear.weight[0][1]
        bias = self.model.linear.bias
        def theta_func(cov, t):
            def softplus(x, B=100):
                return torch.nn.functional.softplus(x, B)
            activation = coef_time * t + coef_cov * cov + bias
            ret =  (softplus(activation) - t).cpu().detach().numpy()[0]
            return ret
        return theta_func

    def sample_single_traj_and_event_time(self, class_idx, max_traj_len=1000):
        cov_vals = [self.cov_traj_func_per_class[class_idx](0)]
        cov_times = [0]
        thetas = [self.theta_func(cov_vals[0], 0)]
        
        proposed_time = np.inf 
        time_delta = self.sample_time_delta_per_class(class_idx)
        i = 0
        continue_loop = (cov_times[-1] + time_delta < proposed_time) and (i < max_traj_len)
        while continue_loop:
            next_cov_time = cov_times[-1] + time_delta
            next_cov_val = self.cov_traj_func_per_class[class_idx](next_cov_time)
            # main difference from SimpleData class
            next_theta = self.theta_func(next_cov_val, next_cov_time)
            
            cov_times.append(next_cov_time)
            cov_vals.append(next_cov_val)
            thetas.append(next_theta)         

            proposed_time = self.sample_proposed_time(
                next_theta, next_cov_time
            )
            time_delta = self.sample_time_delta_per_class(class_idx)

            i += 1
            continue_loop = (cov_times[-1] + time_delta < proposed_time) and (i < max_traj_len)
        if i >= max_traj_len:
            print('Max trajectory length reached while sampling from learned data!')
        true_event_time = proposed_time
        return cov_vals, cov_times, thetas, true_event_time

if __name__ == '__main__':
    main()
    #test_truncated_sampling()
    
