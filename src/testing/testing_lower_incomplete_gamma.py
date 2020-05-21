from utils.loss_calculators import GGDLossCalculator
import torch
from scipy.special import gamma as gamma_func
from scipy.special import gammainc as reg_gamma_func

def test_lower_incomplete_gamma(gamma, x_bound, n_terms=30):
    loss = GGDLossCalculator()
    sklearns_ans = reg_gamma_func(gamma, x_bound)
    print(loss.estimate_lower_incomplete_gamma_with_series(gamma, x_bound, n_terms=n_terms))
    print('sklearn answer', sklearns_ans)

if __name__ == '__main__':
    gamma = torch.tensor(1.9)
    x_bound = torch.tensor(.45)
    n_terms = 30
    test_lower_incomplete_gamma(gamma, x_bound, n_terms=n_terms)
