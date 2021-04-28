import torch
import gc
import numpy as np
from scipy.stats import rankdata
from loss.LossCalculator import LossCalculator


# Note if you have another complicated metric like c-index then you should
# create separate objects for each evaluation type, just update their function
# to call the object and compute the metric
class ModelEvaluator:

    def __init__(self, eval_params, loss_params, model_type, verbose=False):
        self.params = eval_params
        self.verbose = verbose
        self.model_type = model_type
        if not model_type == 'landmarked_cox':
            self.loss_calculator = LossCalculator(loss_params, model_type)

    def evaluate_model(self,
        model, data_input, diagnostics,
        is_during_training=False
    ):
        eval_metrics = {}
        verbose = self.verbose
        
        # free memory since single batch data format used for the rest of code
#        del data_input
#        gc.collect()

        for eval_metric in self.params['eval_metrics']:
            eval_metrics[eval_metric] = {}
            for split in ['tr', 'te']:
#                print('just evaluating over test right now!!! change me back to [tr, te]')
                if split == 'tr':
                    if self.model_type == 'dummy_global_zero_deltas':
                        # model free evaluations use unnormalized data here
                        data = data_input.get_unnormalized_tr_data_as_single_batch() 
                    else:
                        data = data_input.get_tr_data_as_single_batch() 
                else:
                    if self.model_type == 'dummy_global_zero_deltas':
                        # model free evaluations use unnormalized data here
                        data = data_input.get_unnormalized_te_data_as_single_batch() 
                    else:
                        data = data_input.get_te_data_as_single_batch() 
                    
                
                if verbose:
                    print('---------------Evaluating %s-----------------' %eval_metric)
                eval_func_name = 'compute_' + eval_metric 
                if verbose:
                    if split == 'tr':
                        print('Evaluation on Train')
                    else:
                        print('Evaluation on Test')
                        
                metric_results = getattr(self, eval_func_name)(model, data)
                eval_metrics[eval_metric][split] = metric_results 

#        if self.model_type == 'dummy_global_zero_deltas':
#            # model free evaluations use unnormalized data here
#            data_tr = data_input.get_unnormalized_tr_data_as_single_batch() 
#            data_te = data_input.get_unnormalized_te_data_as_single_batch()
#        else:
#            data_tr = data_input.get_tr_data_as_single_batch() 
#            data_te = data_input.get_te_data_as_single_batch()
#        for eval_metric in self.params['eval_metrics']:
#            if verbose:
#                print('---------------Evaluating %s-----------------' %eval_metric)
#            eval_func_name = 'compute_' + eval_metric 
#            if verbose:
#                print('Evaluation on Train')
#            metric_results_tr = getattr(self, eval_func_name)(model, data_tr)
#            if verbose:
#                print('Evaluation on Test')
#            metric_results_te = getattr(self, eval_func_name)(model, data_te)
#            eval_metrics[eval_metric] = {'tr':metric_results_tr, 'te':metric_results_te}
        if is_during_training:
            diagnostics.cur_tracked_eval_metrics = eval_metrics
        else:
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
                if self.verbose:
                    print(dynamic_metrics[s, t], start_time, time_delta)
                doesnt_use_time_delta = \
                    (metric_name == 'c_index_from_start_time') or \
                    (metric_name == 'c_index_from_most_recent_time') or\
                    (metric_name == 'c_index_truncated_at_S')
                if doesnt_use_time_delta:
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
            func = self.compute_c_index_from_start_time_to_event_time_i
        elif metric_name == 'c_index_from_most_recent_time':
            func = self.compute_c_index_from_most_recent_cov_time_k_to_event_time_i
        elif metric_name == 'c_index_truncated_at_S':
            func = self.compute_c_index_truncated_at_S_from_S_to_event_times_i
        elif metric_name == 'auc':
            func = self.compute_auc_at_t_plus_delta_t
        elif metric_name == 'auc_truncated_at_S':
            func = self.compute_auc_truncated_at_S_from_S_to_delta_S
        elif metric_name == 'standard_c_index_truncated_at_S':
            func = self.compute_standard_c_index_truncated_at_S_over_window
        elif metric_name == 'brier_score':
            func = self.compute_brier_score
        elif metric_name == 'd_calibration':
            func = self.compute_d_calibration_with_ind_cens
        else:
            raise ValueError('dynamic metric %s not recognized' %metric_name)
        return func

    ### DEPRECATED!!
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
                    if self.verbose:
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

    def compute_auc_truncated_at_S(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'auc_truncated_at_S') 

    def compute_c_index_truncated_at_S(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'c_index_truncated_at_S')

    def compute_standard_c_index_truncated_at_S(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'standard_c_index_truncated_at_S')

    def compute_c_index(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'c_index')
    
    def compute_c_index_from_start_time(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'c_index_from_start_time')

    def compute_c_index_from_most_recent_time(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'c_index_from_most_recent_time')

    def compute_d_calibration(self, model, split_data):
        return self.evaluate_dynamic_metric(model, split_data, 'd_calibration')

    def compute_auc_truncated_at_S_from_S_to_delta_S(self,
        model, data, start_time, time_delta
    ):
        risks = self.compute_risks(
            model, data,
            start_time, time_delta,
            'auc_truncated_at_S'
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



    def compute_standard_c_index_truncated_at_S_over_window(self,
        model, data, start_time, time_delta
    ):
        risks = self.compute_risks(
            model, data,
            start_time, time_delta,
            'standard_c_index_truncated_at_S'
        ) 
        c_index, tot_valid_pairs = self.calc_standard_c_index_truncated_at_S_with_risks(
            risks, data, start_time, time_delta
        )
        return c_index, tot_valid_pairs

    def calc_standard_c_index_truncated_at_S_with_risks(
        self, risks, data, start_time, time_delta
    ):
        event_in_time_window = \
            (start_time <= data.event_times) &\
            (data.event_times <= start_time + time_delta) &\
            (~data.censoring_indicators.bool())                    

        is_valid = np.zeros((len(risks), len(risks)))
        ordered_correct = np.zeros((len(risks), len(risks)))
        for i in range(len(risks)):
            if not event_in_time_window[i]:
                continue
            time_i = data.event_times[i]
            risk_i = risks[i]        
    
            valid_idxs = np.where(data.event_times > time_i)
            is_valid[i, valid_idxs] = 1
            
            ordered_correct_idxs = np.where(risk_i > risks)
            ordered_correct[i, ordered_correct_idxs] = 1

            tied_idxs = np.where(risk_i == risks)
            ordered_correct[i, tied_idxs] = 1/2

        tot_valid_pairs = np.sum(is_valid)
        if tot_valid_pairs == 0:
            return 0, 0
        c_index = np.sum(is_valid * ordered_correct)/tot_valid_pairs
        return c_index, tot_valid_pairs 

    def compute_c_index_from_most_recent_cov_time_k_to_event_time_i(self,
        model, data, start_time, time_delta=None
    ):
        # time delta not used for this metric
        time_delta = None
        risks_ik = self.compute_risks(
            model, data, 
            start_time, time_delta,
            'c_index_from_most_recent_time'
        )
        total_concordant_pairs, total_valid_pairs = \
            self.compute_c_index_upper_boundary_at_event_time_i(risks_ik)

        if total_valid_pairs == 0:
            # this should only happen for very large values of S
            # where either no one is included, or everyone in the
            # risk set is censored
            return 0, 0
        c_index = total_concordant_pairs/total_valid_pairs
        return c_index, total_valid_pairs


    def compute_c_index_truncated_at_S_from_S_to_event_times_i(self,
        model, data, start_time, time_delta=None
    ):
        # time delta not used for this metric
        time_delta = None
        risks_ik = self.compute_risks(
            model, data, 
            start_time, time_delta,
            'c_index_truncated_at_S'
        )
        total_concordant_pairs, total_valid_pairs = \
            self.compute_c_index_upper_boundary_at_event_time_i(
                risks_ik, data, start_time
            )

        if total_valid_pairs == 0:
            # this should only happen for very large values of S
            # where either no one is included, or everyone in the
            # risk set is censored
            return 0, 0
        c_index = total_concordant_pairs/total_valid_pairs
        return c_index, total_valid_pairs

    def compute_d_calibration_with_ind_cens(self,
        model, data, start_time, time_delta=None,
        num_bins=10
    ):
        surv_probs = \
            self.get_surv_probs_at_end_of_seq(model, data, start_time)

        # compute d-calibration over only the risk set, this is at least the
        # most obvious and intuitive way to do so, although an argument could
        # be made to include all censored individuals, in the risk set or not
        #surv_probs = surv_probs[at_risk_idxs]

        # construct bins
        bin_counts = torch.zeros(num_bins, 1)

        # Evaluate the surv probs on the true uncensored times and add to the counts of dociles of observed times
        at_risk_unc_idxs = \
            (batch.event_times > start_time) & (batch.censoring_indicators == 0)
        surv_probs_unc = surv_probs[at_risk_unc_idxs]
        bin_idxs_unc = torch.floor(surv_probs_unc * (num_bins))
        for b in num_bins:
            bin_counts[b] = bin_counts[b] + torch.sum(bin_idxs_unc == b)

        # Using the survival probs, evaluate the contributions to each bin from the censored observations
        at_risk_cens_idxs = \
            (batch.event_times > start_time) & (batch.censoring_indicators == 1)
        surv_probs_cens = surv_probs[at_risk_cens_idxs]
        # need bin idxs even here since the bin it falls in is treated 
        # differently
        bin_idxs_cens = torch.floor(surv_probs_unc * (num_bins))
        # now compute the 'smoothed' contributions of these censored idxs
        for b in num_bins:
            contr_in_bin = torch.sum(
                1. - (b/num_bins)/surv_probs_cens[bin_idxs_cens == b]
            )
            # CHECK THIS LINE!
            contr_outside_bin = torch.sum(
                1./(num_bins * surv_probs_cens[surv_probs_cens < (b/num_bins)])
            )
        

    def get_surv_probs_at_end_of_seq(self, model, data, start_time):
       
        # get cdf per individual predicted at the given start time
        # (conditioned of course on survival to time t_ij)
        # time delta in this case is a tensor (since deltas are tensors)
        # of the difference tau_i - start time per individual so that the
        # most recent cdf is being computed at the correct times
        deltas, _, _ = model(data)
        eos_minus_start_times = batch.event_times - start_time

        surv_probs = 1. - self.logprob_calculator.compute_most_recent_CDF(\
            deltas, data, model.get_global_param(), 
            start_time, eos_minus_start_times
        )

        return surv_probs 

    def compute_c_index_from_start_time_to_event_time_i(self,
        model, data, start_time, time_delta=None
    ):
        # time delta not used for this metric
        time_delta = None
        risks_ik = self.compute_risks(
            model, data, 
            start_time, time_delta,
            'c_index_from_start_time'
        )
        total_concordant_pairs, total_valid_pairs = \
            self.compute_c_index_upper_boundary_at_event_time_i(
                risks_ik, data, start_time
            )

        if total_valid_pairs == 0:
            # this should only happen for very large values of S
            # where either no one is included, or everyone in the
            # risk set is censored
            if self.verbose:
                print('No valid pairs for start time %d and delta %d' %(start_time, time_delta))
                print('Risks:', risks)
            return 0, 0
        c_index = total_concordant_pairs/total_valid_pairs
        return c_index, total_valid_pairs
        
    def compute_c_index_upper_boundary_at_event_time_i(
        self, risks_ik, data, start_time
    ):
        sorted_event_times, sort_idxs = torch.sort(data.event_times)
        sorted_cens_inds = data.censoring_indicators[sort_idxs].bool()
        total_concordant_pairs = 0
        total_valid_pairs = 0
        for idx_i, risks_k in enumerate(risks_ik):
            # risks_ik is now of shape N X N after bug correction
            # it used to be of shape U X N before bug correction where
            # U was the number of uncensored event times
            # and N is the total number of individuals
            unc_at_risk = (not (sorted_cens_inds[idx_i]) ) and (sorted_event_times[idx_i] > start_time)
            if not unc_at_risk:
                continue
            con_pairs_ik, valid_pairs_ik = \
                self.compute_concordant_and_valid_pairs_ik(
                    idx_i, risks_k
                )
#            print(idx_i, con_pairs_ik, valid_pairs_ik)
#            print(torch.sum(torch.isnan(risks_k)))
            total_concordant_pairs += con_pairs_ik
            total_valid_pairs += valid_pairs_ik
        return total_concordant_pairs, total_valid_pairs        

    def compute_concordant_and_valid_pairs_ik(self, idx_i, risks_k):
        if idx_i == len(risks_k) - 1:
            # this means that this is the last
            # event time since risks_ik is ordered
            # by event times
            return 0, 0
        valid_risks_k = risks_k[idx_i + 1: ]
        risk_i = risks_k[idx_i]
        
        # risk_i > risk_k if T_i < \tau_k
        concordant_bool_idxs = (risk_i - valid_risks_k) > 0
        tied_bool_idxs = risk_i == valid_risks_k
        
        num_concordant = torch.sum(concordant_bool_idxs)
        num_tied = torch.sum(tied_bool_idxs)        
        # ties count as 'half concordant'
        eff_num_concordant = num_concordant + 0.5 * num_tied

        num_valid_pairs = len(valid_risks_k)
        return eff_num_concordant, num_valid_pairs  

#        risks_i = self.compute_risks(
#            model, data, 
#            start_time, time_delta,
#            'c_index_from_start_time'
#        )
#
#        num_individuals = len(data.event_times)
#        num_ordered_correctly = 0
#        normalization = 0
#        
#        valid_bool_idxs_i = \
#            (data.event_times > start_time) &\
#            (~data.censoring_indicators.bool())
#        valid_idxs_i = torch.arange(num_individuals)[valid_bool_idxs_i]
#        
#        for idx_i in valid_idxs_i:
#            valid_idxs_k = torch.arange(
#                num_individuals
#            )[data.event_times > data.event_times[idx_i]]
#            risk_i = risks_i[idx_i]
#            for idx_k in valid_idxs_k:
#                if type(model) == str:
#                    risk_k = risks_i[idx_k]
#                else:
#                    risk_k = self.compute_risk_from_start_time_to_event_time_ik(
#                        model, data, start_time, data.event_times[idx_i], idx_k
#                    )
#                normalization += 1
#                num_ordered_correctly += self.is_ordered_correctly_with_risks_ik(
#                    risk_i, risk_k
#                )
        if normalization == 0:
            return 0
        c_index = num_ordered_correctly/normalization
        return c_index, normalization

    def compute_risk_from_start_time_to_event_time_ik(self,
        model, data, start_time, event_time_i, k
    ):
        pred_params, _, _ = model(data)
        prob_calc = self.loss_calculator.logprob_calculator
        risk_func = prob_calc.compute_cond_prob_from_start_to_event_time_ik
        risk_k = risk_func(
            pred_params, data, model.get_global_param(),
            start_time, event_time_i, k 
        )
        return risk_k
        

    def is_ordered_correctly_with_risks_ik(self, 
        risk_i, risk_k
    ):

        first_risk = risk_i
        second_risk = risk_k

        if first_risk - second_risk > 0:
            return 1
        elif first_risk == second_risk:
            # ties count as 'halfway correct'
            return 0.5
        return 0



    def compute_c_index_at_t_plus_delta_t(self,
        model, data, start_time, time_delta
    ):
        risks = self.compute_risks(
            model, data,
            start_time, time_delta,
            'standard_c_index_truncated_at_S'
        ) 
        c_index, tot_valid_pairs = self.calc_standard_c_index_with_risks(
            risks, data, start_time, time_delta
        )
        return c_index, tot_valid_pairs

    def calc_standard_c_index_with_risks(self,
        risks, data, start_time, time_delta
    ):
        event_in_time_window = \
            (data.event_times <= start_time + time_delta) &\
            (~data.censoring_indicators.bool())                    

        is_valid = np.zeros((len(risks), len(risks)))
        #print(len(risks))
        #print(risks)
        ordered_correct = np.zeros((len(risks), len(risks)))
        for i in range(len(risks)):
            if not event_in_time_window[i]:
                continue
            time_i = data.event_times[i]
            risk_i = risks[i]        
    
            valid_idxs = np.where(data.event_times > time_i)
            is_valid[i, valid_idxs] = 1
            
            ordered_correct_idxs = np.where(risk_i > risks)
            ordered_correct[i, ordered_correct_idxs] = 1

            # ties counts as 1/2
            tied_idxs = np.where(risk_i == risks)
            ordered_correct[i, tied_idxs] = 1/2

        tot_valid_pairs = np.sum(is_valid)
        if tot_valid_pairs == 0:
            if self.verbose:
                print('No valid pairs for start time %d and delta %d' %(start_time, time_delta))
                print('Risks:', risks)
            return 0, 0
        c_index = np.sum(is_valid * ordered_correct)/tot_valid_pairs
        return c_index, tot_valid_pairs 

    def compute_c_index_at_t_plus_delta_t_old(self,
        model, data, start_time, time_delta
    ):
        risks = self.compute_risks(
            model, data, 
            start_time, time_delta,
            'c_index'
        )
        
        num_individuals = len(data.event_times)
        num_ordered_correctly = 0
        normalization = 0
        
        valid_bool_idxs_i = \
            (data.event_times <= start_time + time_delta) &\
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
            return 0, 0
        c_index = num_ordered_correctly/normalization
        return c_index, normalization


    # TODO pull out indexing into the computations or else write a new function for
    # c-index from start time to true event time only
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
                model, data, start_time, time_delta, metric_name,
                rep_for_truncated_c_index=self.params['rep_for_truncated_c_index']
            )
            return risks
        elif type(model).__name__ == 'LandmarkedCoxModel':
            risks = self.compute_landmarked_cox_risks(
                model, data, start_time, metric_name
            )
            return risks

        prob_calc = self.loss_calculator.logprob_calculator
        if metric_name == 'c_index':
            # wasn't carrying forwards the conditioning before, but deephit
            # does actually carry forwards conditioning in their code
#            risk_func = prob_calc.compute_most_recent_CDF
            risk_func = prob_calc.compute_cond_probs_truncated_at_S_over_window
        elif metric_name == 'c_index_from_start_time':
            # in this case we decided it makes most sense to include pairs from S -> infinity

            # instead we get the risks for first slot i here by integrating S to T_i
            risk_func = prob_calc.compute_cond_probs_k_from_start_to_event_times_i
        elif metric_name == 'c_index_from_most_recent_time':
            risk_func = prob_calc.compute_cond_probs_from_cov_time_k_to_event_times_i
        elif metric_name == 'auc_truncated_at_S':
            risk_func = prob_calc.compute_cond_probs_truncated_at_S_over_window
        elif metric_name == 'standard_c_index_truncated_at_S':
            risk_func = prob_calc.compute_cond_probs_truncated_at_S_over_window
        elif metric_name == 'c_index_truncated_at_S':
            risk_func = prob_calc.compute_cond_probs_truncated_at_S_to_event_times_i
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
        model_name, data, start_time, time_delta, metric_name,
        rep_for_truncated_c_index=False
    ):
        if metric_name == 'c_index_truncated_at_S':
            if rep_for_truncated_c_index:
                most_recent_times, most_recent_idxs = \
                    data.get_most_recent_times_and_idxs_before_start(start_time)
                _, sort_idxs = torch.sort(data.event_times)
                # probs should just remove cov times ranking and just have this for
                # num events ranking.. can always add back if needed
                if model_name == 'cov_times_ranking':
                    risks = -most_recent_times
                elif model_name == 'num_events_ranking':
                    risks = most_recent_idxs + 1
                elif model_name == 'framingham':
                    covs_at_time = data.get_unpacked_padded_cov_trajs()[torch.arange(data.event_times.shape[0]), most_recent_idxs, :]
                    #covs_at_time = data.get_unpacked_padded_cov_trajs(most_recent_idxs)
                    risks = self.compute_framingham_total_cvd_risks(data, covs_at_time)
                
                sorted_risks = risks[sort_idxs]
                risks = sorted_risks.repeat([data.event_times.shape[0], 1])
                
            else:
                risks = self.compute_c_index_truncated_at_S_model_independent_risks(
                model_name, data, start_time, time_delta, metric_name
            )
        else:
            most_recent_times, most_recent_idxs = \
                data.get_most_recent_times_and_idxs_before_start(start_time)
            # probs should just remove cov times ranking and just have this for
            # num events ranking.. can always add back if needed
            if model_name == 'cov_times_ranking':
                risks = -most_recent_times
            elif model_name == 'num_events_ranking':
                risks = most_recent_idxs + 1
            elif model_name == 'framingham':
                covs_at_time = data.get_unpacked_padded_cov_trajs()[torch.arange(data.event_times.shape[0]), most_recent_idxs, :]
                #covs_at_time = data.get_unpacked_padded_cov_trajs(most_recent_idxs)
                risks = self.compute_framingham_total_cvd_risks(data, covs_at_time)
            else:
                raise ValueError('model inpendent risks type %s not found' %model_name)
        return risks

    def compute_landmarked_cox_risks(self, model, data, start_time, metric_name):
        _, most_recent_idxs = \
            data.get_most_recent_times_and_idxs_before_start(start_time)
        covs = data.get_unpacked_padded_cov_trajs()
        covs = \
            covs[torch.arange(covs.shape[0]), most_recent_idxs, :]
        covs = covs.cpu().detach().numpy()
        risks = torch.tensor(
            model.models[start_time].predict_risk(covs),
            dtype=torch.float64
        )
        if metric_name == 'c_index_truncated_at_S':
            sorted_risks = risks[torch.sort(data.event_times)[1]]
            risks = sorted_risks.repeat([data.event_times.shape[0], 1])
        return risks
            


    def compute_c_index_truncated_at_S_model_independent_risks(self,
        model_name, data, start_time, time_delta, metric_name
    ):
        risks_ik = self.get_model_independent_risk_matrix(data, model_name)
        _, sort_idxs = torch.sort(data.event_times)
#            risk_matrix = sorted_risks.repeat([data.event_times.shape[0], 1])
        
#        unc_at_risk = \
#            ~data.censoring_indicators[sort_idxs].bool() &\
#            (data.event_times[sort_idxs] > start_time)
#        risks = risks_ik[unc_at_risk]
        risks = risks_ik
        return risks

    # for use with truncated at time t c-index version
    def get_model_independent_risk_matrix(self, data, model_name):
        _, sort_idxs = torch.sort(data.event_times)
        sorted_event_times = data.event_times[sort_idxs]
        risks_ik = torch.zeros(
            sorted_event_times.shape[0], sorted_event_times.shape[0]
        )
        for t, time in enumerate(sorted_event_times):
            if t % 100 == 0:
                #print('...')
                pass
            most_recent_times, most_recent_idxs = \
                data.get_most_recent_times_and_idxs_before_start(time)
            if model_name == 'cov_times_ranking':
                row = -most_recent_times
            elif model_name == 'num_events_ranking':
                row = most_recent_idxs + 1
            elif model_name == 'framingham':
                covs_at_time = data.get_unpacked_padded_cov_trajs()[torch.arange(data.event_times.shape[0]), most_recent_idxs, :]
                row = self.compute_framingham_total_cvd_risks(data, covs_at_time)
            risks_ik[t, :] = row
        return risks_ik

    def compute_framingham_total_cvd_risks(self, data, covs_at_time):
        non_meds_cutoff = 125
        num_meds = 61
        sys_bp_idx = 120
        tc_idx =  44
        hdl_idx = 42
        age_idx = 124
        static_covs = data.static_covs
        age_onset_dm_cvd = data.get_unpacked_padded_cov_trajs()[:, 0, age_idx]
        hypertensive_medicine_indicators =  torch.tensor([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1], dtype=bool).repeat(covs_at_time.shape[0], 1)
#        print(covs_at_time[non_meds_cutoff:non_meds_cutoff + num_meds].shape, 'hiss')
        treated_for_hypertension = torch.sum(covs_at_time[:, non_meds_cutoff: non_meds_cutoff + num_meds].bool() & hypertensive_medicine_indicators, dim=1) > 0
        treated_for_hypertension = treated_for_hypertension.int()
        smoking = static_covs[:, 14]
        is_male = static_covs[:, 1]
#        print(covs_at_time[0:100, age_idx])
#        print(covs_at_time[:, tc_idx])
#        print(covs_at_time[0:100, hdl_idx])
#        print(covs_at_time[:, sys_bp_idx])

        # Risk for males
        S10=0.88936
        S5=(S10**(0.1))**5
        Age=48.5
        TC=212.5
        HDL=44.9
        SBP=129.7
        Hypertension=0.1013
        Smoker=0.3522
        DM=0.065
        m=3.06117*np.log(Age) + 1.12370*np.log(TC)-0.93263*np.log(HDL)+1.99881*np.log(SBP)*(Hypertension) + 1.93303*np.log(SBP)*(1-Hypertension) +0.65451*Smoker + 0.57367*DM
#        print(age_onset_dm_cvd.shape, treated_for_hypertension.shape, smoking.shape, covs_at_time[:, hdl_idx].shape)
        L = \
            3.06117*torch.log(age_onset_dm_cvd) + \
            1.12370*torch.log(covs_at_time[:, tc_idx]) - \
            0.93263*torch.log(covs_at_time[:, hdl_idx]) + \
            1.99881*torch.log(covs_at_time[:, sys_bp_idx]) * (treated_for_hypertension)+ \
            1.93303*torch.log(covs_at_time[:, sys_bp_idx]) * (1 - treated_for_hypertension)+ \
            0.65451 * smoking + 0.57367
        Risk_male = (1-S5**(torch.exp(L-m))) 
        
        # Risk for females
        S10=0.95012
        S5=(S10**(0.1))**5
        Age=49.1
        TC=215.1
        HDL=57.6
        SBP=125.8
        Hypertension=0.1176
        Smoker=0.3423
        DM=0.0376
        m=2.32888*np.log(Age) + 1.20904*np.log(TC)-0.70833*np.log(HDL)+2.76157*np.log(SBP)*(Hypertension) + 2.82263*np.log(SBP)*(1-Hypertension) +0.52873*Smoker + 0.69154*DM
        L = \
            2.32888*torch.log(age_onset_dm_cvd) + \
            1.20904*torch.log(covs_at_time[:, tc_idx]) - \
            0.70833*torch.log(covs_at_time[:, hdl_idx]) + \
            2.76157*torch.log(covs_at_time[:, sys_bp_idx])*(treated_for_hypertension) + \
            2.82263*torch.log(covs_at_time[:, sys_bp_idx])*(1 - treated_for_hypertension) + \
            0.52873 * smoking + 0.69154
        Risk_female = (1-S5**(torch.exp(L-m))) 
        Risks = is_male * Risk_male + (1 - is_male) * Risk_female
        return Risks 
       
