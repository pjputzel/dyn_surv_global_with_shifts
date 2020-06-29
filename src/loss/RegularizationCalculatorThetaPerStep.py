import torch
import torch.nn as nn

class RegularizationCalculatorThetaPerStep(nn.Module):
    
    def __init__(self, loss_params):
        super().__init__()
        self.params = loss_params
    
    def forward(self,
        global_theta, pred_params, hidden_states,
        step_ahead_cov_preds, batch, ret_each_term=False
    ):

        step_ahead_cov_reg = 0
        theta_drift_reg = 0
        global_diff_reg = 0

        if not self.params['step_ahead_cov_reg'] == 0:
            step_ahead_cov_reg = self.compute_step_ahead_cov_reg(
                step_ahead_cov_preds, batch
            )

        if not self.params['theta_drift_reg'] == 0:
            theta_drift_reg = self.compute_theta_drift_reg(
                pred_params, batch
            )

        if not self.params['global_diff_reg'] == 0:  
            global_diff_reg = self.compute_global_diff_reg(
                pred_params, global_theta
            )


        step_ahead_cov_reg *= self.params['step_ahead_cov_reg']
        theta_drift_reg *= self.params['theta_drift_reg']
        global_diff_reg *= self.params['global_diff_reg']

        if ret_each_term:
            return step_ahead_cov_reg, theta_drift_reg, global_diff_reg

        return step_ahead_cov_reg + theta_drift_reg + global_diff_reg

    # TODO:mask out missing values here as in Dynamic DeepHit
    def compute_step_ahead_cov_reg(self, step_ahead_cov_preds, batch):

        batch_cov_trajs = batch.get_unpacked_padded_cov_trajs()       
        diffs_squared_averages_per_individual = torch.zeros(batch_cov_trajs.shape[0])
        iterations = enumerate(
            zip(
                batch_cov_trajs, step_ahead_cov_preds,
                batch.trajectory_lengths.int()
            )
        )
        for i, (batch_cov_traj, next_step_cov_pred, length) in iterations:
            
            if length == 1:
                continue
            diffs_squared_averages_per_individual[i] = torch.mean(
                (batch_cov_traj[1:length, 1:] - next_step_cov_pred[0:length - 1])**2
            )


        mean = torch.mean(diffs_squared_averages_per_individual)
        return mean



    def compute_theta_drift_reg(self, pred_params, batch):
        # TODO: add 1/(delta_t) as a prefactor
        diffs = (pred_params[:, :pred_params.shape[1] - 1] - pred_params[:, 1:])**2
        return torch.mean(torch.mean(diffs))

    def compute_global_diff_reg(self, pred_params, global_theta):
        dist_from_global = (pred_params/global_theta - 1)**2
        return torch.mean(torch.mean(dist_from_global))
