import torch
import math
import numpy as np
import time

def print_grad(grad):
    print('gradient in trainer:', grad)


class BasicModelTrainer:
    def __init__(self, train_params):
        self.params = train_params
        self.max_iter = 10
    def train_model(self, model, data_input, diagnostics, loss_type='total_loss'):
        start_time = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params['learning_rate'])
        # TODO add options for training type-either convergence or fixed epochs
        # for now its until convergence
        # TODO figure out what the tracker should do here- maybe just have the model output
        # relevant metrics in a dict and the tracker just saves them
        prev_loss = torch.tensor(0)
        cur_loss = torch.tensor(np.inf)
        cur_log_loss = 0
        epoch = 0
        while torch.abs(cur_loss - prev_loss) > self.params['conv_thresh'] and epoch < self.max_iter:
            data_input.prepare_sequences_for_rnn(self.params['batch_size'])
            prev_loss = cur_loss
            diagnostics_per_batch, cur_loss = self.step_params_over_all_batches(model, data_input, optimizer, diagnostics, loss_type=loss_type)
            diagnostics.update_full_data_diagnostics(diagnostics_per_batch, epoch)
            cur_log_loss = diagnostics.full_data_diagnostics['loss']
            if epoch % self.params['n_epoch_print'] == 0:
                diagnostics.print_cur_diagnostics()
            epoch += 1
        #update tracker one last time
        if not epoch - 1 == diagnostics.epochs[-1]:
            diagnostics.update_full_data_diagnostics(diagnostics_per_batch. epoch)
            diagnostics.print_cur_diagnostics()
        print('Total training time for loss type %s was %d seconds' %(loss_type, time.time() - start_time))
        return diagnostics

    def  step_params_over_all_batches(self, model, data_input, optimizer, diagnostics, next_step_reg_str=.1, loss_type='total_loss'):
        # note this function will eventually need to include concatentation with the missing data indicators
        #print(len(data_input.batches_of_padded_sequences), self.params['batch_size'])
        cur_diagnostics_per_batch = []
        for batch_idx in range(len(data_input.batches_of_padded_sequences)):
            optimizer.zero_grad()
            cur_batch_covs = data_input.batches_of_padded_sequences[batch_idx]
            
            batch_event_times = torch.tensor(data_input.event_times[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])

            batch_censoring_indicators = torch.tensor(data_input.censoring_indicators[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
            cur_diagnostics = diagnostics.compute_batch_diagnostics(model, cur_batch_covs, batch_event_times, batch_censoring_indicators)#model(cur_batch)
           
            if loss_type == 'total_loss':
                cur_loss = cur_diagnostics['total_loss']
            elif loss_type == 'reg_only':
                cur_loss = cur_diagnostics['regularization']
            elif loss_type == 'log_loss_only':
                cur_loss = cur_diagnostics['loss']
            else:
                raise ValueError('Loss type %s for training model not recognized' %loss_type)
            #if use_MLE:
            #    pred_distribution_params = pred_distribution_params + self.get_MLE(data_input, model.distribution_type)
            # move to the diagnostics object
            #loss = self.compute_batch_exp_distribution_loss(data_input, pred_distribution_params, batch_idx)
            #loss = loss + next_step_reg_str * self.compute_next_step_cov_loss(cur_batch, next_step_cov_preds)
            #cur_diagnostics['total_loss'].backward()
            cur_loss.backward()
            
            optimizer.step()
            cur_diagnostics_per_batch.append(cur_diagnostics)
        return cur_diagnostics_per_batch, cur_loss

    def compute_next_step_cov_loss(self, cur_batch, next_step_cov_preds):
        batch_covs, _ = torch.nn.utils.rnn.pad_packed_sequence(cur_batch) 
        next_step_cov_preds = torch.tensor(next_step_cov_preds)
     #   print(next_step_cov_preds.shape)
     #   print(batch_covs[1:next_step_cov_preds.shape[0] + 1].shape, 'batch cov shape')
        # normalize to make the magnitude dataset invariant
        diffs_squared = (batch_covs[1:len(next_step_cov_preds) + 1] - torch.tensor(next_step_cov_preds))**2
        mean = torch.mean(diffs_squared)
        std = diffs_squared.std()
        normalized_diffs = (diffs_squared - mean)/std
        
        return torch.sum(normalized_diffs)

    def compute_batch_exp_distribution_loss(self, data_input, pred_distribution_params, batch_idx):
        #data_input.prepare_event_times_for_loss_computation()
        batch_event_times = torch.tensor(data_input.event_times[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        logpdf = self.compute_batch_exp_distribution_logpdf(batch_event_times, pred_distribution_params)
        survival_logprobs = self.compute_batch_exp_distribution_log_survival_probabilities(batch_event_times, pred_distribution_params)
        #print(torch.sum(torch.isnan(survival_logprobs)), torch.sum(torch.isinf(survival_logprobs)))
        #print(torch.exp(survival_logprobs))
        #print(torch.exp(logpdf))
        batch_censoring_indicators = torch.tensor(data_input.censoring_indicators[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        
        #print('logpdf', logpdf)
        #print('survival_logprobs', survival_logprobs)
        #print(batch_censoring_indicators)
        loss_per_individual = logpdf * batch_censoring_indicators +  survival_logprobs * (1 - batch_censoring_indicators)
        #print(logpdf)
        #if loss_per_individual > 0:
        #    print('meow?????')
        #    print('loss', loss_per_individual)
        #    print('logpdf', logpdf)
        #    print('surv', survival_logprobs)
        #    print('params', pred_distribution_params)
        #    print('event times', batch_event_times)
        return -1./(loss_per_individual.shape[0]) * torch.sum(loss_per_individual)

    def compute_batch_exp_distribution_logpdf(self, batch_event_times, pred_distribution_params):
        return torch.log(pred_distribution_params[:, 0]) - pred_distribution_params[:, 0] * batch_event_times

    def compute_batch_exp_distribution_log_survival_probabilities(self, batch_event_times, pred_distribution_params):
        return -pred_distribution_params[:, 0] * batch_event_times
    
    def compute_batch_lnormal_distribution_loss(self, data_input, pred_distribution_params, batch_idx):
        #data_input.prepare_event_times_for_loss_computation()
        batch_event_times = torch.tensor(data_input.event_times[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        logpdf = self.compute_batch_lnormal_distribution_logpdf(batch_event_times, pred_distribution_params)
        survival_logprobs = self.compute_batch_lnormal_distribution_log_survival_probabilities(batch_event_times, pred_distribution_params)
        #print(torch.sum(torch.isnan(survival_logprobs)), torch.sum(torch.isinf(survival_logprobs)))
        #print(torch.exp(survival_logprobs))
        #print(torch.exp(logpdf))
        batch_censoring_indicators = torch.tensor(data_input.censoring_indicators[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        
        #print('logpdf', logpdf)
        #print('survival_logprobs', survival_logprobs)
        #print(batch_censoring_indicators)
        loss_per_individual = logpdf * batch_censoring_indicators +  survival_logprobs * (1 - batch_censoring_indicators)
        #print(logpdf)
        #if loss_per_individual > 0:
        #    print('meow?????')
        #    print('loss', loss_per_individual)
        #    print('logpdf', logpdf)
        #    print('surv', survival_logprobs)
        #    print('params', pred_distribution_params)
        #    print('event times', batch_event_times)
        return -1./(loss_per_individual.shape[0]) * torch.sum(loss_per_individual)



    def compute_batch_lnormal_distribution_logpdf(self, batch_event_times, pred_distribution_params):
        mu = pred_distribution_params[:, 0]
        sigma = pred_distribution_params[:, 1]
        #term1 = torch.log(1./(batch_event_times * sigma * torch.log(torch.tensor(2.  * math.pi))))
        term1 = -(torch.log(batch_event_times) + torch.log(sigma) + (1./2.) * torch.log(torch.tensor(2. * math.pi)))
        #term1 = -(torch.log(batch_event_times) + torch.log(sigma * (2. * math.pi)**.5))
        #print(torch.max(-torch.log(batch_event_times)), torch.max(-torch.log(sigma)), -(1./2.) * torch.log(torch.tensor(2. * math.pi)))
        #print('sigmam min', torch.min(sigma))
        #print('mu min, mu max', torch.min(mu), torch.min(mu)) 
        term2 = -((torch.log(batch_event_times) - mu)**2.)/(2. * sigma**2.)
        ##print('max terms in logpdf', torch.max(term1), torch.max(term2))
        #print('max sum t1 and t2', torch.max(term1 + term2))

        #term1 = -(torch.log(batch_event_times) + torch.log(batch_event_times)**2/(2. * sigma**2) - mu * torch.log(batch_event_times)/(sigma**2))
        #term2 = -(torch.log(sigma * (2. * math.pi)**.5) + mu**2/(2. * sigma**2.))
        #print('max terms in logpdf', torch.max(term1), torch.max(term2))
        #print('max sum t1 and t2', torch.max(term1 + term2))
        return term1 + term2

    def compute_batch_lnormal_distribution_log_survival_probabilities(self, batch_event_times, pred_distribution_params):
        mu = pred_distribution_params[:, 0]
        sigma = pred_distribution_params[:, 1]

        erf_arg = (torch.log(batch_event_times) - mu)/(2**(1./2.) * sigma)
        erf = torch.erf(erf_arg)

        cdf = .5 + .5 * erf
        return torch.log(1. - cdf)

        

    def compute_batch_gamma_distribution_loss(self, data_input, pred_distribution_params, batch_idx):
        #data_input.prepare_event_times_for_loss_computation()
        batch_event_times = torch.tensor(data_input.event_times[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        logpdf = self.compute_batch_gamma_distribution_logpdf(batch_event_times, pred_distribution_params)
        survival_logprobs = self.compute_batch_gamma_distribution_log_survival_probabilities(batch_event_times, pred_distribution_params)
        #print(torch.sum(torch.isnan(survival_logprobs)), torch.sum(torch.isinf(survival_logprobs)))
        #print(torch.exp(survival_logprobs))
        #print(torch.exp(logpdf))
        batch_censoring_indicators = torch.tensor(data_input.censoring_indicators[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        
        #print('logpdf', logpdf)
        #print('survival_logprobs', survival_logprobs)
        #print(batch_censoring_indicators)
        # double check that your censoring indicators are 1 if censored
        loss_per_individual = logpdf * batch_censoring_indicators +  survival_logprobs * (1 - batch_censoring_indicators)
        #print(logpdf)
        if loss_per_individual > 0:
            print('meow?????')
            print('loss', loss_per_individual)
            print('logpdf', logpdf)
            print('surv', survival_logprobs)
            print('params', pred_distribution_params)
        return -1./(loss_per_individual.shape[0]) * torch.sum(loss_per_individual)


    def compute_batch_gamma_distribution_logpdf(self, batch_event_times, pred_distribution_params):
        alpha = pred_distribution_params[:, 0]
        beta = pred_distribution_params[:, 1]
        pdf = beta**alpha/(torch.exp(torch.lgamma(alpha))) * batch_event_times**(alpha - 1) * torch.exp(-beta * batch_event_times)
        # there are numerical issues with torch.lgamma :(
        log_pdf = torch.clamp(torch.log(pdf), max=0)
        #log_pdf2 = alpha * torch.log(beta) - torch.lgamma(alpha) + (alpha - 1) * torch.log(batch_event_times) - beta * batch_event_times
        #print('normal', alpha, beta)
        #print(batch_event_times)
        #if log_pdf > 0:
        #    print(log_pdf, log_pdf2)
        #    print('????????????????????????????????????????????????????????????????????????????????????????????', alpha, beta, log_pdf)
        return log_pdf

    def compute_batch_gamma_distribution_log_survival_probabilities(self, batch_event_times, pred_distribution_params, epsilon=1e-5):
        alpha = pred_distribution_params[:, 0]
        beta = pred_distribution_params[:, 1] 
        gamma_cdf = torch.clamp(self.estimate_regularized_lower_incomplete_gamma_with_series(alpha, batch_event_times * beta), max=1. - epsilon)
        #print(gamma_cdf)
        #print(torch.log(1. - gamma_cdf))
        return torch.log(1. - gamma_cdf)


    # might make sense to refactor these into their own distribution object            
    def compute_batch_GGD_loss(self, data_input, pred_distribution_params, batch_idx):
        # insert equation here with pdf from survival analysis results
        #data_input.prepare_event_times_for_loss_computation()
        batch_event_times = torch.tensor(data_input.event_times[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        logpdf = self.compute_batch_GGD_logpdf(batch_event_times, pred_distribution_params)
        survival_logprobs = self.compute_batch_GGD_log_survival_probabilities(batch_event_times, pred_distribution_params)
        #print(torch.sum(torch.isnan(survival_logprobs)), torch.sum(torch.isinf(survival_logprobs)))
        #print(torch.exp(survival_logprobs))
        #print(torch.exp(logpdf))
        batch_censoring_indicators = torch.tensor(data_input.censoring_indicators[batch_idx * self.params['batch_size']: (batch_idx + 1) * self.params['batch_size']])
        
        #print('logpdf', logpdf)
        #print('survival_logprobs', survival_logprobs)
        loss_per_individual = logpdf * batch_censoring_indicators +  survival_logprobs * (1 - batch_censoring_indicators)
        #print(loss_per_individual)
        return -1./(loss_per_individual.shape[0]) * torch.sum(loss_per_individual)
        
   
    def compute_batch_GGD_logpdf(self, event_times, pred_distribution_params):
        lambda_param = pred_distribution_params[:, 0]
        beta_param = pred_distribution_params[:, 1]
        sigma_param = pred_distribution_params[:, 2]
        #print('lambda', lambda_param)
        #print('beta', beta_param)
        #print('sigma', sigma_param)
        
        #print(sigma_param) 
        #part1 = torch.log(torch.abs(lambda_param)/(sigma_param * event_times * torch.exp(torch.lgamma(lambda_param**(-2) ))))
        part1 = torch.log(torch.abs(lambda_param)) - torch.log(sigma_param * event_times) - torch.lgamma(lambda_param**(-2))
        #print(torch.sum(torch.isnan(part1)))
        
        #print(lambda_param**-2)
        part2_1 = torch.log((lambda_param ** -2)) * (lambda_param ** -2)
        part2_2 = torch.log(torch.exp(-beta_param) * event_times) * (1./(lambda_param * sigma_param))
        #part2 = torch.log(( lambda_param**-2  * (torch.exp(-beta_param) * event_times)**(lambda_param**-2)) )
        part2 = part2_1 + part2_2
        
        #part3 = torch.log(torch.exp(-lambda_param**-2 * (torch.exp(-beta_param) * event_times)**(lambda_param/sigma_param)))
        #print((torch.exp(-beta_param) * event_times)**(lambda_param/sigma_param)) 
        part3 = -lambda_param**-2 * (torch.exp(-beta_param) * event_times)**(lambda_param/sigma_param)
        part3 = torch.clamp(part3, -1e8)
        #print('PART1', part1)
        #print('PART2', part2)
        #print('lambda param', lambda_param)
        #print('beta param', beta_param)
        #print('sigma_param', sigma_param)
        
        #print('PART3', part3)
        #print('Sum of parts', part1+part2+part3)
        #print(torch.exp(part1 + part2 + part3))
        return part1 + part2 + part3
 
    def compute_batch_GGD_log_survival_probabilities(self, event_times, pred_distribution_params, epsilon=1e-10, max_integral_range=1e10, min_logprob=-1e8):
        # torch doesn't implement the cdf of the gamma function so we have to estimate it by using the gamma function and integrating. Note that we are following Cox 2007 for computing the Survival function using a lower incomplete gamma function
        lambda_params = pred_distribution_params[:, 0]
        beta_params = pred_distribution_params[:, 1]
        sigma_params = pred_distribution_params[:, 2]
        #pdf_gamma = gamma.log_pdf(event_times) 
        # now estimate the lower incomplete gamma function
        survival_logprobs = torch.zeros(event_times.shape[0])
        x_boundaries = (lambda_params**-2 * ((torch.exp(-beta_params) * event_times)**(lambda_params/sigma_params)))
        x_boundaries = torch.clamp(x_boundaries, epsilon, max_integral_range)
        gamma_concentrations = lambda_params ** (-2)
        lower_incomplete_gammas = self.estimate_lower_incomplete_gamma_with_series(gamma_concentrations, x_boundaries)
        return torch.clamp(torch.log((lambda_params > 0) * (1 - lower_incomplete_gammas + epsilon) + (lambda_params <= 0) * (lower_incomplete_gammas + epsilon)), min=min_logprob)
        #for i, (lambd, beta, sigma, event_time) in enumerate(zip(lambda_params, beta_params, sigma_params, event_times)):
        #    gamma = torch.distributions.gamma.Gamma(concentration=lambd**-2, rate=1)

        #    x_boundary = (lambd**-2 * ((torch.exp(-beta) * event_time)**(lambd/sigma))).item()
        #    if x_boundary > max_integral_range:
        #        x_boundary = max_integral_range
        #    #print(x_boundary, 'x_boundary')
        #    integral_xrange = torch.linspace(0, x_boundary, steps=10000000)
        #    integral_xrange[0] = integral_xrange[0] + epsilon
        #    pdf_gamma = torch.exp(gamma.log_prob(integral_xrange))
        #    #print(integral_xrange)
        #    #print('pdf gamma', pdf_gamma)
        #    prob = torch.trapz(pdf_gamma, integral_xrange)
        #    #print('est lower incomplete gamma: ', prob)
        #    probs.append(prob) 
        #    if lambd > 0:
        #        prob = torch.clamp(prob, 0., 1. - epsilon)
        #        if prob >= 1:
        #            #prob = torch.clamp(prob, 0., 1. - epsilon)
        #            #prob = prob - (prob - 1. +  epsilon)
        #            #print('meow', prob, lambd)
        #            pass
        #        survival_logprobs[i] = torch.log(1 - prob)
        #        #print(prob, torch.log(1 - prob), 'labmda >0 case')
        #    else:
        #        prob = torch.clamp(prob, epsilon, 1.)
        #        if prob <= 0:
        #            #prob = prob - (prob - epsilon)
        #            #prob = torch.clamp(prob, epsilon, 1.)
        #            pass
        #        survival_logprobs[i] = torch.log(prob)
                #print(prob, torch.log(prob), 'lambda <=0 case')
        #print(survival_logprobs)
        #return torch.clamp(survival_logprobs, min_logprob, 0.)

    # maybe just use stirlings approximation
    def estimate_regularized_lower_incomplete_gamma_with_series(self, concentration, upper_limit, rate=1., n_terms=150, upper_clamp=1e8):
        sum_idxs = torch.arange(n_terms).view(-1, 1)
        #print(concentration.shape)
        #print(sum_idxs.shape)
        #fix_shapes(sum_idxs, concentration)
        #normalization = torch.lgamma(concentration)
        # since we are normalized ignore the numerator in the series
        term1 = -torch.lgamma(concentration + sum_idxs + 1.)
        term2 = torch.log(upper_limit) * sum_idxs
        #print('term1', torch.sum(torch.isinf(term1)))
        #print('term1', torch.sum(torch.isnan(term1)))
        #print('term2', torch.sum(torch.isnan(term2)))
        #print('term2', torch.sum(torch.isinf(term2)))        
        #print('term2, upper_limit', term2[torch.isnan(term2)], upper_limit[torch.isnan(term2)[0, :]])
        #print('term2', term2)

        term_per_sum_idx = torch.clamp(torch.exp(term1 + term2), 0., upper_clamp)

        #print('term per sum idx', torch.sum(torch.isnan(term_per_sum_idx)))
        #print('term_per_sum_idx', torch.sum(torch.isinf(term_per_sum_idx)))
        
        prefactor = torch.clamp(torch.exp(torch.log(upper_limit) * concentration - upper_limit), max=1e8)
        #print('prefactor', torch.sum(torch.isnan(prefactor)))
        #print('prefactor', torch.sum(torch.isinf(prefactor)))
        series_estimate = torch.clamp(prefactor * torch.sum(term_per_sum_idx, dim=0), 0., 1.)

        #print('series estimate', torch.sum(torch.isnan(series_estimate)))
        #print('series estimate', torch.sum(torch.isnan(series_estimate)))
        #print('series estimate', torch.sum(torch.isinf(series_estimate)))
        #print('series estimate', series_estimate[torch.isnan(series_estimate)], torch.exp(prefactor)[torch.isnan(series_estimate)])
        #print(torch.exp(log_prefactor))
        #print('series estimate', series_estimate)
        return series_estimate

