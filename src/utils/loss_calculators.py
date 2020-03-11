import torch
import numpy as np


### Helper Functions         
def estimate_lower_reg_incomplete_gamma_with_series(gamma_concentration, x_boundary, n_terms=3):
    #gamma_concentration.register_hook(print_grad)
    # from gammainc.pdf in bookmarks
    #gamma_concentration.register_hook(print_grad)
    x_boundary_lengths = [x[~(x == 0)].shape[0] for x in x_boundary]
    #print('lengths', x_boundary_lengths)
    if len(x_boundary.shape) > 1:
        prefactor = torch.zeros(x_boundary.shape[0], np.max(x_boundary_lengths))
        #print(prefactor.shape, x_boundary.shape, gamma_concentration.shape)
        for i, length in enumerate(x_boundary_lengths):
#            print(prefactor[i, :length].shape, x_boundary[i, :length].shape, gamma_concentration.shape)
            prefactor[i, :length] = x_boundary[i, :length] ** gamma_concentration[i] * torch.exp(-x_boundary[i, :length])
    else:
        prefactor = x_boundary**gamma_concentration * torch.exp(-x_boundary) #* torch.as_tensor(~(x_boundary == 0), dtype=torch.double)
    # this is to not take derivatives wrt to the padding in the cov times during computation of 'normalization term' in loss
    #x_boundary.register_hook(replace_nans_with_zero)
    #print(prefactor)
    #print(x_boundary)
    
    #prefactor.register_hook(print_grad)
    #x_boundary.register_hook(print_grad)
    #gamma_concentration.register_hook(print_grad)
    #print(x_boundary.shape, prefactor.shape, gamma_concentration.shape)
    #prefactor.register_hook(print_grad)
    #print(prefactor.shape, 'prefactor')
    sum_of_terms = 0
    for i in range(n_terms):
        denominator = gamma_concentration
        for j in range(i):
            denominator = denominator * (gamma_concentration + j + 1)
#            print('denominators', denominator, i, j)
#            denominator.register_hook(print_grad)
        #if i == 0:
        #    denominator.register_hook(print_grad)
            #pass
            
        term_i = x_boundary**i / denominator
        #term_i.register_hook(print_grad)
        #print(denominator , i, 'denom')
        sum_of_terms = sum_of_terms + term_i
    #sum_of_terms.register_hook(print_grad)
    #print(sum_of_terms.shape)
    #print(sum_of_terms.shape, 'sum of terms')
    return prefactor * sum_of_terms/torch.exp(torch.lgamma(gamma_concentration)) * torch.as_tensor(~(x_boundary == 0), dtype=torch.double)

# hooks for debugging and gradient clamping
def print_grad(grad):
    print('gradient', grad)
def replace_nans_with_zero(grad):
    grad[torch.isnan(grad)] = 0
    return grad

class BaseLossCalculator:

    def compute_loss(self, model_outputs):
        raise NotImplementedError('Must subclass BaseLossCalculator and overwrite compute loss function')

class ExponentialLossCalculator(BaseLossCalculator): 

    def compute_loss(self, batch_event_times, batch_cov_times, batch_traj_lengths, pred_distribution_params, batch_censoring_indicators):

        batch_traj_lengths = torch.tensor(batch_traj_lengths, dtype=torch.float64)

        logpdf = self.compute_batch_exp_distribution_logpdf(pred_distribution_params, batch_event_times)

        survival_logprobs = self.compute_batch_exp_distribution_log_survival_probabilities(pred_distribution_params, batch_event_times)
        
        #print(logpdf.shape, survival_logprobs.shape, batch_censoring_indicators.shape)

        #batch_censoring_indicators = batch_censoring_indicators.unsqueeze(1)
        #batch_cov_final_times = torch.zeros(batch_cov_times.shape[0])
        #for i, length in enumerate(batch_traj_lengths):
        #    batch_cov_final_times[i] = batch_cov_times[i, length - 1]
            #print(batch_cov_times[i, length - 1])
        #print(batch_cov_times[:, batch_traj_lengths - 1].shape)
        #print(batch_traj_lengths.shape)

        #print(logpdf.shape)
        # subtract batch_traj weights so large sequences don't eat up all the loss
        loss_per_individual = batch_traj_lengths * (logpdf * (1 - batch_censoring_indicators) +  survival_logprobs * (batch_censoring_indicators))


        # Normalization to account for forwards prediction from time t
        #print(self.compute_batch_exp_distribution_log_survival_probabilities(pred_distribution_params, batch_cov_times))
        # correct this term for where the sequences end by looking at where batch_cov_times is zero/whatever
        normalization_term = - torch.sum(self.compute_batch_exp_distribution_log_survival_probabilities(pred_distribution_params, batch_cov_times, is_cov_times=True), dim=1)
        #print(loss_per_individual.shape, normalization_term.shape)
        return -1. * torch.mean(loss_per_individual + normalization_term)

    def compute_batch_exp_distribution_logpdf(self, pred_distribution_params, batch_event_times):
        pred_distribution_params = pred_distribution_params.unsqueeze(1)
        #print(pred_distribution_params.shape, batch_event_times.shape)
        return ((torch.log(pred_distribution_params[:]) - pred_distribution_params[:] * batch_event_times.unsqueeze(1))).squeeze(1)
    
    def compute_batch_exp_distribution_log_survival_probabilities(self, pred_distribution_params, batch_event_times, is_cov_times=False):
        if not is_cov_times:
            batch_event_times = batch_event_times.unsqueeze(1)
        pred_distribution_params = pred_distribution_params.unsqueeze(1)
        #print(batch_event_times.shape)
            
   #     print(pred_distribution_params.shape,  batch_event_times.shape)
        return ((-pred_distribution_params[:] * batch_event_times) * torch.as_tensor(~(batch_event_times == 0), dtype=torch.double)).squeeze(1)

    #def compute_batch_exp_distribution_logpdf(self, batch_event_time_deltas, pred_distribution_params, batch_event_times):

    #    #print(pred_distribution_params.shape, batch_event_time_deltas.shape, batch_event_times.shape)
    #    survival_logprobs = -pred_distribution_params[:, :, 0] * (batch_event_times.unsqueeze(1) - batch_event_time_deltas)
    #    return ((torch.log(pred_distribution_params[:, :, 0]) - pred_distribution_params[:, :, 0] * batch_event_time_deltas)  - survival_logprobs) * (~(batch_event_time_deltas == batch_event_times.unsqueeze(1)))
    #
    #def compute_batch_exp_distribution_log_survival_probabilities(self, batch_event_time_deltas, pred_distribution_params, batch_event_times):
   ##     print(pred_distribution_params.shape, batch_event_time_deltas.shape, batch_event_times.shape)
    #    return (-pred_distribution_params[:, :, 0] * batch_event_time_deltas) * (~(batch_event_time_deltas == batch_event_times.unsqueeze(1)))


class GammaLossCalculator:
    
    def compute_loss(self, batch_event_times, batch_cov_times, batch_traj_lengths, pred_distribution_params, batch_censoring_indicators):
        logpdf = self.compute_batch_gamma_logpdf(pred_distribution_params, batch_event_times)
   #     logpdf.register_hook(print_grad)

        survival_logprobs = self.compute_batch_gamma_log_survival_probabilities(pred_distribution_params, batch_event_times)
        #survival_logprobs.register_hook(print_grad)
        batch_traj_lengths = torch.as_tensor(batch_traj_lengths, dtype=torch.double)
        loss_per_individual = batch_traj_lengths * (logpdf * (1 - batch_censoring_indicators) +  survival_logprobs * (batch_censoring_indicators)) 
        #loss_per_individual.register_hook(print_grad)

        # Normalization to account for forwards prediction from time t
        # correct this term for where the sequences end by looking at where batch_cov_times is zero/whatever
        
        # uncomment while seeing if we are summing over padded indices incorrectly
        #print(self.compute_batch_gamma_log_survival_probabilities(pred_distribution_params, batch_cov_times, compute_over_sequence=True), 'meowowww')
        normalization_term = - torch.sum(self.compute_batch_gamma_log_survival_probabilities(pred_distribution_params, batch_cov_times, compute_over_sequence=True), dim=1)
#        normalization_term.register_hook(print_grad)
        #print(normalization_term.shape, loss_per_individual.shape)
        #print(normalization_term[torch.isnan(normalization_term)])
        #print(normalization_term)
        full_loss = -1. * torch.mean(loss_per_individual + normalization_term)
        #full_loss.register_hook(print_grad)
        return full_loss

    def compute_batch_gamma_logpdf(self, pred_distribution_params, batch_event_times):
        alpha = pred_distribution_params[:, 0]
        #alpha.register_hook(print_grad)
        beta = pred_distribution_params[:, 1]
        #beta.register_hook(print_grad)
        #print(alpha[0], beta[0])
        #pdf = beta**alpha/(torch.exp(torch.lgamma(alpha))) * batch_event_times**(alpha - 1) * torch.exp(-beta * batch_event_times)
        log_pdf = alpha * torch.log(beta) - torch.lgamma(alpha) + (alpha - 1) * torch.log(batch_event_times) - beta * batch_event_times
        #log_pdf.register_hook(print_grad)
        #print(log_pdf, 'log pdf directly')
        #print(torch.log(pdf), 'log pdf indirectly')
        #print(torch.lgamma(alpha))
        # there are numerical issues with torch.lgamma :(
        #log_pdf = torch.log(pdf)
        #print('pdf:', log_pdf.shape)
#        log_pdf = torch.clamp(log_pdf, max=0)
        return log_pdf



    def compute_batch_gamma_log_survival_probabilities(self, pred_distribution_params, batch_event_times, compute_over_sequence=False):
        if compute_over_sequence:
            alpha = pred_distribution_params[:, 0].unsqueeze(1)
            beta = pred_distribution_params[:, 1].unsqueeze(1)
        else:
            alpha = pred_distribution_params[:, 0]
            beta = pred_distribution_params[:, 1]
        #alpha.register_hook(print_grad)
        #beta.register_hook(print_grad)
        #print(beta.shape, batch_event_times.shape)
        gamma_cdf  = estimate_lower_reg_incomplete_gamma_with_series(alpha, batch_event_times * beta)
        #gamma_cdf = torch.clamp(self.estimate_regularized_lower_incomplete_gamma_with_series(alpha, batch_event_times * beta), max=1. - epsilon)
        #gamma_cdf.register_hook(print_grad)
#        print('survival prob:', 1. - gamma_cdf)
#        print('log survival prob:', torch.log(1. - gamma_cdf))
#        print('batch event times', batch_event_times)
        #print(torch.as_tensor((batch_event_times == 0), dtype=torch.double))
        return (torch.log(1. - gamma_cdf) * torch.as_tensor(~(batch_event_times == 0), dtype=torch.double))

class GGDLossCalculator:
    
    def compute_loss(self, batch_event_times, batch_cov_times, batch_traj_lengths, pred_distribution_params, batch_censoring_indicators):

        #print(pred_distribution_params)
        logpdf = self.compute_batch_GGD_distribution_logpdf(pred_distribution_params, batch_event_times)

        survival_logprobs = self.compute_batch_GGD_distribution_log_survival_probabilities(pred_distribution_params, batch_event_times)
        
        #print(logpdf.shape, survival_logprobs.shape, batch_censoring_indicators.shape)

        #batch_censoring_indicators = batch_censoring_indicators.unsqueeze(1)
        #batch_cov_final_times = torch.zeros(batch_cov_times.shape[0])
        #for i, length in enumerate(batch_traj_lengths):
        #    batch_cov_final_times[i] = batch_cov_times[i, length - 1]
            #print(batch_cov_times[i, length - 1])
        #print(batch_cov_times[:, batch_traj_lengths - 1].shape)
        #print(batch_traj_lengths.shape)
        batch_traj_lengths = torch.as_tensor(batch_traj_lengths, dtype=torch.double)
        loss_per_individual = batch_traj_lengths * (logpdf * (1 - batch_censoring_indicators) +  survival_logprobs * (batch_censoring_indicators)) 

        # Normalization to account for forwards prediction from time t
        # correct this term for where the sequences end by looking at where batch_cov_times is zero/whatever
        
        # uncomment while seeing if we are summing over padded indices incorrectly
        #print(self.compute_batch_GGD_distribution_log_survival_probabilities(pred_distribution_params, batch_cov_times, compute_over_sequence=True).shape)
        normalization_term = - torch.sum(self.compute_batch_GGD_distribution_log_survival_probabilities(pred_distribution_params, batch_cov_times, compute_over_sequence=True), dim=1)
        #print(normalization_term.shape, loss_per_individual.shape)
        #print(normalization_term[torch.isnan(normalization_term)])
        #print(normalization_term)
        full_loss = -1. * torch.mean(loss_per_individual + normalization_term)
        return full_loss

    def compute_batch_GGD_distribution_logpdf(self, pred_distribution_params, batch_event_times):

        lambda_param = pred_distribution_params[:, 0]
        beta_param = pred_distribution_params[:, 1]
        sigma_param = pred_distribution_params[:, 2]
        part1 = torch.log(torch.abs(lambda_param)) - torch.log(sigma_param * batch_event_times) - torch.lgamma(lambda_param**(-2))

   #     print(pred_distribution_params.shape,  batch_event_times.shape)

        part2_1 = torch.log((lambda_param ** -2)) * (lambda_param ** -2)
        part2_2 = torch.log(torch.exp(-beta_param) * batch_event_times) * (1./(lambda_param * sigma_param))


        part2 = part2_1 + part2_2
        
        part3 = -lambda_param**-2 * (torch.exp(-beta_param) * batch_event_times)**(lambda_param/sigma_param)

        #print(part1[torch.isnan(part1)], part2[torch.isnan(part2)], part3[torch.isnan(part3)])
        return part1 + part2 + part3
    
    def compute_batch_GGD_distribution_log_survival_probabilities(self, pred_distribution_params, batch_event_times, compute_over_sequence=False):
        if compute_over_sequence:
            lambda_params = pred_distribution_params[:, 0].unsqueeze(1)
            beta_params = pred_distribution_params[:, 1].unsqueeze(1)
            sigma_params = pred_distribution_params[:, 2].unsqueeze(1)
        else:
            lambda_params = pred_distribution_params[:, 0]
            beta_params = pred_distribution_params[:, 1]
            sigma_params = pred_distribution_params[:, 2]
        #sigma_params.register_hook(print_grad)    
        # now estimate the lower incomplete gamma function
        survival_logprobs = torch.zeros(batch_event_times.shape[0])
        #print(batch_event_times.shape, lambda_params.shape)
        x_boundaries = (lambda_params**-2 * ((torch.exp(-beta_params) * batch_event_times)**(lambda_params/sigma_params)))
        #print(x_boundaries[torch.isinf(x_boundaries)])
#        x_boundaries = torch.clamp(x_boundaries, epsilon, max_integral_range)
        gamma_concentrations = lambda_params ** (-2)
        #print(x_boundaries[x_boundaries < 0])
        #print(gamma_concentrations[gamma_concentrations < 0]) 
        lower_reg_incomplete_gammas = estimate_lower_reg_incomplete_gamma_with_series(gamma_concentrations, x_boundaries)
        lower_reg_incomplete_gammas.register_hook(print_grad)
        #print(lower_incomplete_gammas.shape)
        #print(lower_incomplete_gammas[:, 0])
        #print('is less than zero in survival function?', (1 - lower_incomplete_gammas)[1 - lower_incomplete_gammas < 0])
        #print('is zero in survival function?', (1 - lower_incomplete_gammas)[1 - lower_incomplete_gammas == 0])
        return torch.log(1 - lower_reg_incomplete_gammas) * torch.as_tensor(~(batch_event_times == 0), dtype=torch.double)
 #       return (-pred_distribution_params[:, 0] * batch_event_times) * ~(batch_event_times == 0)


class RegularizationCalculator:

    def __init__(self, regularization_params):
        self.params = regularization_params

    def compute_regularization(self, cur_batch_covs, cur_diagnostics):
        return self.params['next_step_cov_reg_str'] * self.compute_next_step_cov_loss(cur_batch_covs, cur_diagnostics) + self.params['parameter_diversity_reg_str'] * self.compute_parameter_diversity_loss(cur_diagnostics)

    def compute_next_step_cov_loss(self, cur_batch, cur_diagnostics):
        predicted_distribution_params = cur_diagnostics['predicted_distribution_params']
        next_step_cov_preds = cur_diagnostics['next_step_cov_preds']
        batch_covs, lengths = torch.nn.utils.rnn.pad_packed_sequence(cur_batch)
        #print(batch_covs.transpose(0, 1))
        batch_covs = batch_covs.transpose(0,1)
        #print(batch_covs[0, :20, 1:])
        #print(next_step_cov_preds[0, 0:20])
        #print(batch_covs.shape)
        #next_step_cov_preds = torch.tensor(next_step_cov_preds)
        # normalize to make the magnitude dataset invariant
        #batch_covs_for_loop = batch_covs.reshape([batch_covs.shape[1], batch_covs.shape[0], -1])
        #next_step_cov_preds_for_loop = next_step_cov_preds.reshape([next_step_cov_preds.shape[1], next_step_cov_preds.shape[0], -1])
        diffs_squared_averages_per_individual = torch.zeros(batch_covs.shape[0])
        for i, (batch_cov, next_step_cov_pred, length) in enumerate(zip(batch_covs, next_step_cov_preds, lengths)):
            #print('preds', next_step_cov_pred[0:length -1])
            #print('covs', batch_cov[1:length])
            if length == 1:
                continue
            #diffs_squared_averages_per_individual[i] = torch.mean((batch_cov[1:length] - next_step_cov_pred[0:length - 1])**2)
            diffs_squared_averages_per_individual[i] = torch.mean((batch_cov[1:length, 1:] - next_step_cov_pred[0:length - 1])**2)


        #diffs_squared = (batch_covs[1:len(next_step_cov_preds) + 1] - next_step_cov_preds)**2
        mean = torch.mean(diffs_squared_averages_per_individual)
        std = diffs_squared_averages_per_individual.std()
        if std == 0:
            normalized_diffs = torch.abs(diffs_squared_averages_per_individual - mean)
        else:
            normalized_diffs = torch.abs(diffs_squared_averages_per_individual - mean)/std
        #print('std', std, 'mean', mean)
        if torch.sum(~(normalized_diffs == mean/std)) == 0:
            return torch.tensor(0.)
        else:
            #print(torch.sum(~(normalized_diffs == mean/std)))
            #print(torch.mean(normalized_diffs[~(normalized_diffs == mean/std)]))
            if std == 0:
                return_val = torch.mean(normalized_diffs[~(normalized_diffs == mean)])
            else:
                return_val =  torch.mean(normalized_diffs[~(normalized_diffs == mean/std)])

        if torch.isnan(return_val):
            print('std', std, 'mean', mean)
        #return_val = 1/(batch_covs.shape[1] * batch_covs.shape[0] * batch_covs.shape[2]) * torch.sum(diffs_squared_averages_per_individual)
        return_val = mean
        return return_val

    def compute_parameter_diversity_loss(self, cur_diagnostics):
        cur_parameters = cur_diagnostics['predicted_distribution_params']
        #lengths = cur_diagnostics['sequence_lengths']
        #most_recent_param_preds = torch.zeros(cur_parameters.shape[0], cur_parameters.shape[1])
        #for i, length in enumerate(lengths):
        #    most_recent_param_preds[i, :] = cur_parameters[i, length - 1]
        
        #std = most_recent_param_preds.std()
        #std = torch.sum(cur_parameters.std(dim=0))
        #print(std)
        #return std
        return 0
            
