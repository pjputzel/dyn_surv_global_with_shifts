import torch
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
            metric_tr = getattr(self, eval_func_name)(model, data_tr)
            metric_te = getattr(self, eval_func_name)(model, data_te)
            eval_metrics[eval_metric] = {'tr':metric_tr, 'te':metric_te}

   
    def compute_c_index(self, model, split_data):
        # TODO include handling predictions from last available
        # event time using a negative one -not sure if this makes sense....

        start_times = self.params['c_index']['start_times']
        window_length = self.params['c_index']['window_length']
        time_step = self.params['c_index']['time_step']
        num_time_steps = int(window_length//time_step)
        time_deltas = [time_step * (i + 1) for i in range(int(num_time_steps))]
    
        c_indices = -1 * torch.ones(len(start_times), num_time_steps)
        for s, start_time in enumerate(start_times):
            for t, time_delta in enumerate(time_deltas):
                c_indices[s, t] = self.compute_c_index_at_t_plus_delta_t(
                    model, split_data, start_time, time_delta
                )

                print(c_indices[s, t], start_time, time_delta, len(split_data.event_times))
                

        ret = {\
            'start_times': start_times, 
            'time_deltas': time_deltas, 
            'values':c_indices
        }
        return ret


    def compute_c_index_at_t_plus_delta_t(self,
        model, data, start_time, time_delta
    ):
        risks = self.compute_risks(
            model, data, 
            start_time, time_delta
        )
        print('one risk', risks[0])
        print('another', risks[1])
        print('and a third', risks[2])
         

        num_individuals = len(data.event_times)
        num_ordered_correctly = 0
        normalization = 0
        for i in range(num_individuals):
            if data.censoring_indicators[i]:
                continue
            # event time i must be within t_ij* and t_ij* + \delta_p
            # for now hardcoded to work with baseline model, update later
            most_recent_cov_time_i = 0.
            in_interval = \
                (time_delta + most_recent_cov_time_i >= data.event_times[i])\
                and (data.event_times[i] >= most_recent_cov_time_i)

            if not in_interval:
                continue

            for j in range(num_individuals):
                #TODO: figure out if there needs to be a different
                # condition for \tau_j
                # looks like no extra condition is correct as long as \tau_j > \tau_i
                # we're gtg 
                #if start_time >= data.event_times[j]:
                #    continue
                # again this is hardcoded now for the baseline model, update later
                most_recent_cov_time_j = 0
                if data.event_times[j] > most_recent_cov_time_j + time_delta:
                    normalization += 1
                    num_ordered_correctly += self.is_ordered_correctly(
                        risks, i, j
                    )
        if normalization == 0:
            return 0
        print('num_ordered_correctly:', num_ordered_correctly, 'normalization:', normalization)
        return num_ordered_correctly/normalization

    def is_ordered_correctly(self, 
        risks,  first_idx, second_idx
    ):

        first_risk = risks[first_idx]
        second_risk = risks[second_idx]

        if first_risk - second_risk > 0:
            return 1
        return 0
        
        
    def compute_risks(self,
        model, data,
        start_time, time_delta,
    ):
        #TODO: add to logprob calculator a compute_conditional_CDF function
        # which takes in the start time and finish time
        pred_params, _, _ = model(data)
        logprob_calculator = self.loss_calculator.logprob_calculator
        risks = logprob_calculator.compute_cond_prob_over_window(
            pred_params, data, model.get_global_param(),
            start_time, time_delta
        )

        return risks 
