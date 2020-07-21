import torch
import torch.nn as nn

class RegularizationCalculatorDeltaIJ(nn.Module):
    
    def __init__(self, loss_params):
        super().__init__()
        self.params = loss_params
    
    def forward(self,
        global_theta, pred_params, hidden_states,
        step_ahead_cov_preds, batch, ret_each_term=False
    ):

        delta_reg = 0

        if not self.params['delta_reg'] == 0:  
            delta_reg = self.compute_delta_reg(
                pred_params, global_theta
            )


        delta_reg = self.params['delta_reg'] * delta_reg

        if ret_each_term:
            return delta_reg

        return delta_reg



    def compute_delta_reg(self, pred_params, global_theta):
        delta_reg = (pred_params)**2
        return torch.mean(torch.mean(delta_reg))
