import torch
import numpy as np
from scipy.stats import rankdata
from loss.LossCalculator import LossCalculator


# Note if you have another complicated metric like c-index then you should
# create separate objects for each evaluation type, just update their function
# to call the object and compute the metric
class ModelEvaluator:

    def __init__(self, eval_params, loss_params, model_type):
        self.params = eval_params
        self.loss_calculator = LossCalculator(loss_params, model_type)

    def evaluate_model(self, model, data_input, diagnostics):
        eval_metrics = {}
        for eval_metric in self.params['eval_metrics']:
            eval_func_name = 'compute_' + eval_metric
            data_tr = data_input.get_tr_data_as_single_batch() 
            data_te = data_input.get_te_data_as_single_batch()
            print('Evaluation on Train')
            metric_results_tr = getattr(self, eval_func_name)(model, data_tr)
            print('Evaluation on Test')
            metric_results_te = getattr(self, eval_func_name)(model, data_te)
            eval_metrics[eval_metric] = {'tr':metric_results_tr, 'te':metric_results_te}
        diagnostics.eval_metrics = eval_metrics


    def evaluate_dynamic_metric(self, model, split_data, metric_name):
        start_times = self.params['dynamic_metrics']['start_times']
        window_length = self.params['dynamic_metrics']['window_length']
        time_step = self.params['dynamic_metrics']['time_step']
        num_time_steps = int(window_length//time_step)
        time_deltas = [time_step * (i + 1) for i in range(int(num_time_steps))]
    
        dynamic_metrics = -1 * torch.ones(len(start_times), num_time_steps)
        eff_ns = []
        for s, start_time in enumerate(start_times):
            eff_ns_s = []
            for t, time_delta in enumerate(time_deltas):
                eval_func = self.get_dynamic_metric_func(metric_name)
                dynamic_metrics[s, t], eff_n = eval_func(
                    model, split_data, start_time, time_delta
                )
                eff_ns_s.append(eff_n)    
                print(dynamic_metrics[s, t], start_time, time_delta)
                if metric_name == 'c_index_from_start_time':
                    # this metric is independent of the time delta
                    dynamic_metrics[s, :] = dynamic_metrics[s, t]
                    for i in range(len(time_deltas) - 1):
                        eff_ns_s.append(eff_n)
                    break
            eff_ns.append(eff_ns_s)
        ret = {\
            'start_times': start_times, 
            'time_deltas': time_deltas, 
            'values': dynamic_metrics,
            'eff_n': eff_ns
        }
        return ret

    def get_dynamic_metric_func(self, metric_name):
        if metric_name == 'c_index':
            func = self.compute_c_index_at_t_plus_delta_t
        elif metric_name == 'c_index_from_start_time':
            func = self.compute_c_index_from_start_time_to_infinity
        elif metric_name == 'auc':
            func = self.compute_auc_at_t_plus_delta_t
        else:
            raise ValueError('dynamic metric %s not recognized' %metric_name)
        return func

    def evaluate_dynamic_metric_over_num_event_matched_groups(self, model, data, metric_name):
        all_groups_res = {} 

        start_times = self.params['dynamic_metrics']['start_times']
        window_length = self.params['dynamic_metrics']['window_length']
        time_step = self.params['dynamic_metrics']['time_step']
        max_num_bins = self.params['dynamic_metrics']['max_num_bins']
        num_time_steps = int(window_length//time_step)
        time_deltas = [time_step * (i + 1) for i in range(int(num_time_steps))]
    
        dynamic_metrics = -1 * torch.ones(len(start_times), max_num_bins, num_time_steps)
        eff_ns = []
        bin_boundaries = torch.zeros(len(start_times), max_num_bins)
        for s, start_time in enumerate(start_times):
            eff_ns_s = []
            matched_groups, upper_bin_boundaries = \
                data.split_into_binned_groups_by_num_events(
                    max_num_bins, start_time
                )
            groups_iterator = enumerate(zip(matched_groups, upper_bin_boundaries))
            for g, (group, upper_bin_boundary) in groups_iterator:
                eff_ns_s_g = []
#                all_groups_res['events_bin=' + str(num_events)] = group_res
                bin_boundaries[s][g] = upper_bin_boundary
                for t, time_delta in enumerate(time_deltas):
                    eval_func = self.get_dynamic_metric_func(metric_name)
                    dynamic_metrics[s, g, t], eff_n = eval_func(
                        model, group, start_time, time_delta
                    )
                    eff_ns_s_g.append(eff_n)
                    print(dynamic_metrics[s, g, t], start_time, upper_bin_boundary, time_delta)
                eff_ns_s.append(eff_ns_s_g)
            eff_ns.append(eff_ns_s)


        all_groups_res = {\
            'bin_upper_boundaries': bin_boundaries,
            'start_times': start_times, 
            'time_deltas': time_deltas, 
            'values': dynamic_metrics,
            'eff_n': eff_ns
        }

        return all_groups_res
            
        
    def compute_auc_grouped_by_num_events(self, model, split_data):
        ret = self.evaluate_dynamic_metric_over_num_event_matched_groups(
            model, split_data, 'auc'
        )
        return ret

    def compute_c_index_grouped_by_num_events(self, model, split_data):
        ret = self.evaluate_dynamic_metric_over_num_event_matched_groups(
            model, split_data, 'c_index'
        )
        return ret

    def compute_auc(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'auc')
 
    def compute_c_index(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'c_index')
    
    def compute_c_index_from_start_time(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'c_index_from_start_time')

    def compute_auc_at_t_plus_delta_t(self,
       model, data, start_time, time_delta
    ):
        risks = self.compute_risks(
            model, data, 
            start_time, time_delta,
            'auc'
        )
######## For [S, S + \Delta S] version
        case_bool_idxs = \
            (start_time <= data.event_times) &\
            (data.event_times <= start_time + time_delta) &\
            (~data.censoring_indicators.bool())                    
        num_cases = torch.sum(case_bool_idxs)
        control_bool_idxs = data.event_times > start_time + time_delta
        num_controls = torch.sum(control_bool_idxs)
        all_valid_bool_idxs = case_bool_idxs | control_bool_idxs
        
        case_risks = risks[case_bool_idxs]
        control_risks = risks[control_bool_idxs]
        all_risks = torch.cat([case_risks, control_risks])
        ranks = rankdata(all_risks.cpu().detach().numpy())        
        
        #for computing what happens if we flip the order       
#        maxranks = np.max(ranks)
#        flipped_ranks = maxranks + 1 - ranks
                

        case_ranks = ranks[0:num_cases]
        # compute mann-whitney U test statistic
        # and then get the auc from it by normalizing
        U =  np.sum(case_ranks) - ((num_cases) * (num_cases + 1))/2.
        #print('R/U/cases/controls %.2f/%.2f/%d/%d' %(np.sum(case_ranks), U, num_cases, num_controls))
        return U/(num_cases * num_controls), {'cases':num_cases, 'controls':num_controls}



    def compute_c_index_from_start_time_to_infinity(self,
        model, data, start_time, time_delta=None
    ):
        # time delta not used for this metric
        time_delta = None
        risks = self.compute_risks(
            model, data, 
            start_time, time_delta,
            'c_index_from_start_time'
        )

        num_individuals = len(data.event_times)
        num_ordered_correctly = 0
        normalization = 0
        
        valid_bool_idxs_i = \
            (data.event_times > start_time) &\
            (~data.censoring_indicators.bool())
        valid_idxs_i = torch.arange(num_individuals)[valid_bool_idxs_i]
        
        for idx_i in valid_idxs_i:
            valid_idxs_k = torch.arange(
                num_individuals
            )[data.event_times > data.event_times[idx_i]]

            for idx_k in valid_idxs_k:
                normalization += 1
                num_ordered_correctly += self.is_ordered_correctly(
                    risks, idx_i, idx_k
                )
        if normalization == 0:
            return 0
        c_index = num_ordered_correctly/normalization
        return c_index, normalization

    def compute_c_index_at_t_plus_delta_t(self,
        model, data, start_time, time_delta
    ):
        risks = self.compute_risks(
            model, data, 
            start_time, time_delta,
            'c_index'
        )
        #print('one risk', risks[0])
        #print('another', risks[1])
        #print('and a third', risks[2])

#        bool_idxs_less_than_start = data.cov_times <= start_time
#        print(bool_idxs_less_than_start.shape,' bool ids less than start')
#        idxs_most_recent_times = torch.max(torch.where(
#            bool_idxs_less_than_start,
#            data.cov_times, torch.zeros(data.cov_times.shape)
#        ), dim=1)[1]
#        print(idxs_most_recent_times.shape, 'idxs most recent times')
#        most_recent_times = data.cov_times[torch.arange(idxs_most_recent_times.shape[0]), idxs_most_recent_times]
        #print(most_recent_times[0:20], start_time)
#        print(most_recent_times.shape, 'most recent times shape')
######## For [S, S + \Delta S] version
        # May want to use this function still?
#        ranks = rankdata(all_risks.cpu().detach().numpy())        
        
        num_individuals = len(data.event_times)
        num_ordered_correctly = 0
        normalization = 0
        
        valid_bool_idxs_i = \
            (data.event_times <= start_time + time_delta) &\
            (~data.censoring_indicators.bool())
        valid_idxs_i = torch.arange(num_individuals)[valid_bool_idxs_i]
        
        for idx_i in valid_idxs_i:
#            if data.censoring_indicators[i]:
#                continue
##            most_recent_cov_time_i = \
##                torch.max(data.cov_times[i, data.cov_times[i] <= start_time])
#            in_interval = \
#                ( start_time + time_delta >= data.event_times[i])
##                (data.event_times[i] >= start_time)
#
#            if not in_interval:
#                continue
#
            valid_idxs_k = torch.arange(
                num_individuals
            )[data.event_times > data.event_times[idx_i]]

            for idx_k in valid_idxs_k:
                normalization += 1
                num_ordered_correctly += self.is_ordered_correctly(
                    risks, idx_i, idx_k
                )
                
#            for j in range(num_individuals):
#                #TODO: figure out if there needs to be a different
#                # condition for \tau_j
#                # looks like no extra condition is correct as long as \tau_j > \tau_i
#                # we're gtg 
#                #if start_time >= data.event_times[j]:
#                #    continue
#                if data.event_times[j] > data.event_times[i]:
#                    normalization += 1
#                    num_ordered_correctly += self.is_ordered_correctly(
#                        risks, i, j
#                    )
        if normalization == 0:
            return 0
        #print('num_ordered_correctly:', num_ordered_correctly, 'normalization:', normalization)
        c_index = num_ordered_correctly/normalization
        print(normalization)
        return c_index, normalization



    def is_ordered_correctly(self, 
        risks,  first_idx, second_idx
    ):

        first_risk = risks[first_idx]
        second_risk = risks[second_idx]

        if first_risk - second_risk > 0:
            return 1
        elif first_risk == second_risk:
            # ties count as 'halfway correct'
            return 0.5
        return 0
        
        
    def compute_risks(self,
        model, data,
        start_time, time_delta,
        metric_name
    ):
        if type(model) == str :
            # for model independent evaluation
            risks = self.compute_model_independent_risk(
                model, data, start_time, time_delta
            )
            return risks

        prob_calc = self.loss_calculator.logprob_calculator
        if metric_name == 'c_index':
            # in this case use the current CDF
            risk_func = prob_calc.compute_most_recent_CDF
        elif metric_name == 'c_index_from_start_time':
            # in this case we decided it makes most sense to go from S -> infinity
            # Ie survival function should be used here
            risk_func = prob_calc.compute_survival_probability
        else:
            # all other cases integrate from S to S + \Delta S
            risk_func = prob_calc.compute_cond_prob_over_window

        pred_params, _, _ = model(data)
        risks = risk_func(
            pred_params, data, model.get_global_param(),
            start_time, time_delta
        )
        return risks 

    def compute_model_independent_risk(self, 
        model_name, data, start_time, time_delta
    ):
        most_recent_times, most_recent_idxs = \
            data.get_most_recent_times_and_idxs_before_start(start_time)
        if model_name == 'cov_times_ranking':
            risks = most_recent_times
        elif model_name == 'num_events_ranking':
            risks = most_recent_idxs + 1
        return risks
