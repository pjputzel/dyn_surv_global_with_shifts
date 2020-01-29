import torch
import numpy as np
import time

class VanillaModelTrainer:
    def __init__(self, train_params):
        self.params = train_params
    
    def train_model(self, model, data_input, tracker):
        start_time = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'] 
        # TODO add options for training type-either convergence or fixed epochs
        # for now its until convergence
        prev_loss = torch.tensor(0)
        cur_loss = torch.tensor(np.inf)
        epoch = 0
        data_input.prepare_sequences_for_rnn(self.params['batch_size'])
        while torch.abs(cur_loss - prev_loss) > train_params['conv_thresh']:
            prev_loss = cur_loss
            cur_loss = self.step_params_over_all_batches(model, data_input, optimizer)
            if epoch % train_params['n_epoch_eval'] == 0:
                tracker.update(epoch)
                tracker.print_cur_metrics()
            epoch += 1
        #update tracker one last time
        if not epoch - 1 == tracker.epochs[-1]:
            tracker.update(epoch)

    def  step_params_over_all_batches(self, model, data_input, optimizer):
        # note this funciton will eventually need to include concatentation with the missing data indicators
        for batch_idx in range(self.params['batch_size']):
            optimizer.zero_grad()
            cur_batch = data_input.batches_of_padded_sequences[batch_idx]
            pred_distribution_params = model(cur_batch)
            loss = self.compute_GGD_loss(data_input, pred_distribution_params)
            loss.backwards()

    # might make sense to refactor these into their own distribution object            
    def compute_GGD_loss(self, data_input, pred_distribution_params):
        # insert equation here with pdf from survival analysis results
        data_input.prepare_event_times_for_loss_computation()
        pdf = self.compute_GGD_pdf(data_input.event_times, pred_distribution_params)
        survival_probs = self.compute_GGD_survival_probabilities(data_input.event_times, pred_distribution_params)
        
        loss_per_individual = pdf**censoring_indicators * survival_probs**(1 - censoring_indicators)
        return 1/loss_per_individual.shape[0] * torch.sum(loss_per_individual)
        
   
    def compute_GGD_pdf(self, event_times, pred_distribution_params):
        lambda_param = pred_distribution_params[0]
        beta_param = pred_distribution_params[1]
        sigma_param = pred_distribution_params[2]
        
        part1 = torch.abs(lambda_param)/(sigma_param * event_times * torch.exp(torch.log_gamma(lambda_param**-2))
        part2 = ( lambda_param**-2  * (torch.exp(-beta_param) * event_times)**(lambda_param**-2))
        part3 = torch.exp(-labmda**-2 * (torch.exp(-beta_param) * event_times)**(lambda_param/sigma_param))

        return part1 * part2 * part3
 
    def compute_GGD_survival_probabilities(self, event_times, pred_distribution_params):
        # torch doesn't implement the cdf of the gamma function so we have to estimate it by using the gamma function and integrating. Note that we are following Cox 2007 for computing the Survival function using a lower incomplete gamma function
        lambda_param = pred_distribution_params[0]
        beta_param = pred_distribution_params[1]
        sigma_param = pred_distribution_params[2]
    
        gamma = torch.distributions.gamma.Gamma(concentration=lambda_param**-2, scale=1)
        #pdf_gamma = gamma.log_pdf(event_times)
        
        # now estimate the lower incomplete gamma function
        integral_xranges = [torch.linspace(0, lambda_param**-2 * torch.exp(-beta_param) * event_time)**(lambda_param/sigma_param) for event_time in event_times]
        pdf_gammas = [gamma.log_pdf(integral_xrange) for integral_xrange in integral_xranges]
        survival_probs = [torch.trapz(pdf_gamma, integral_xrange) for integral_xrange in integral_xranges]
        if lambda_param > 0:
            return 1 - survival_probs
        return survival_probs

    # TODO: move to data input
    def prepare_sequences_for_rnn(self, covariate_trajectories, missing_indicators):
        # first we need to pad the sequences to the max trajectory length
        padded_trajectories, trajectory_lengths = self.pad_data_with_zeros(covariate_trajectories)

        padded_trajectories = torch.tensor(padded_trajectories)
        trajectory_lengths = torch.tensor(trajectory_lengths)
        batches_of_padded_sequences = []
        for batch in range((padded_trajectories.shape[0]//self.params['batch_size']) + 1):
            batch_input = padded_trajectories[batch * self.params['batch_size']: (batch + 1) * self.params['batch_size']]
            batch_trajectory_lengths = trajectory_lengths[batch * self.params['batch_size']: (batch + 1) * self.params['batch_size']]
            batch_input = batch_input.view(padded_trajectories.shape[1], self.params['batch_size'], 1 if len(padded_trajectories.shape) == 2 else padded_trajectories.shape[2])
            rnn_input = torch.nn.pack_padded_sequence(batch_input, batch_trajectory_lengths)
            batches_of_padded_sequences.append(batch)
        return batches_of_padded_sequences
        
    
    def pad_data_with_zeros(self, covariate_trajectories): 
        max_len_trajectory = np.max([len(traj) for traj in covariate_trajectories])
        padded_trajectories = []
        trajectory_lengths = []
        for traj in covariate_trajectories:
            if len(traj) < max_len_trajectory:
                padded_trajectory = traj + [0 for i in range(max_len_trajectory) - len(traj)]
            padded_trajectories.append(padded_trajectory)
            trajectory_lengths.append(len(traj))
        return padded_trajectories, trajectory_lengths
