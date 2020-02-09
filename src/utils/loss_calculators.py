import torch

class BaseLossCalculator:

    def compute_loss(self, model_outputs):
        raise NotImplementedError('Must subclass BaseLossCalculator and overwrite compute loss function')

class ExponentialLossCalculator(BaseLossCalculator): 
    def compute_loss(self, batch_event_times, pred_distribution_params, batch_censoring_indicators):
        logpdf = self.compute_batch_exp_distribution_logpdf(batch_event_times, pred_distribution_params)

        survival_logprobs = self.compute_batch_exp_distribution_log_survival_probabilities(batch_event_times, pred_distribution_params)

        loss_per_individual = logpdf * (1 - batch_censoring_indicators) +  survival_logprobs * (batch_censoring_indicators)

        return -1./(loss_per_individual.shape[0]) * torch.sum(loss_per_individual)

    def compute_batch_exp_distribution_logpdf(self, batch_event_times, pred_distribution_params):
        return torch.log(pred_distribution_params[:, 0]) - pred_distribution_params[:, 0] * batch_event_times
    
    def compute_batch_exp_distribution_log_survival_probabilities(self, batch_event_times, pred_distribution_params):
        return -pred_distribution_params[:, 0] * batch_event_times



class RegularizationCalculator:

    def __init__(self, regularization_params):
        self.params = regularization_params

    def compute_regularization(self, cur_batch_covs, cur_diagnostics):
        return self.params['next_step_cov_reg_str'] * self.compute_next_step_cov_loss(cur_batch_covs, cur_diagnostics)

    def compute_next_step_cov_loss(self, cur_batch, cur_diagnostics):
        predicted_distribution_params = cur_diagnostics['predicted_distribution_params']
        next_step_cov_preds = cur_diagnostics['next_step_cov_preds']

        batch_covs, lengths = torch.nn.utils.rnn.pad_packed_sequence(cur_batch)
        #print(batch_covs.transpose(0, 1))
        batch_covs = batch_covs.transpose(0,1)
        #print(batch_covs.shape)
        #next_step_cov_preds = torch.tensor(next_step_cov_preds)
        # normalize to make the magnitude dataset invariant
        #batch_covs_for_loop = batch_covs.reshape([batch_covs.shape[1], batch_covs.shape[0], -1])
        #next_step_cov_preds_for_loop = next_step_cov_preds.reshape([next_step_cov_preds.shape[1], next_step_cov_preds.shape[0], -1])
        diffs_squared_averages_per_individual = torch.zeros(batch_covs.shape[0])
        for i, (batch_cov, next_step_cov_pred, length) in enumerate(zip(batch_covs, next_step_cov_preds, lengths)):
            #print(next_step_cov_pred[0:length -1])
            #print(batch_cov[1:length])
            if length == 1:
                continue
            #diffs_squared_averages_per_individual[i] = torch.mean((batch_cov[1:length] - next_step_cov_pred[0:length - 1])**2)
            diffs_squared_averages_per_individual[i] = torch.mean((batch_cov[1:length, 1:] - next_step_cov_pred[0:length - 1, 1:])**2)

        #diffs_squared = (batch_covs[1:len(next_step_cov_preds) + 1] - next_step_cov_preds)**2
        mean = torch.mean(diffs_squared_averages_per_individual)
        std = diffs_squared_averages_per_individual.std()
        normalized_diffs = torch.abs(diffs_squared_averages_per_individual - mean)/std
        return torch.mean(normalized_diffs[~(normalized_diffs == mean/std)])
