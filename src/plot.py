from main_types.DeltaPerStepResultsPlottingMain import DeltaPerStepResultsPlottingMain
from utils.ParameterParser import ParameterParser
import sys
sys.path.append("..")

sys.path.append('../data/COVID-19/')
from preprocess_data import COVID19_Preprocessor


if __name__ == '__main__':
#    path_to_config = '../configs/linear_baseline_configs/covid_configs/linear_delta_per_step_covid.yaml'
    #path_to_config = '../configs/linear_baseline_configs/covid_configs/learn_fixed_theta_linear_delta_per_step_covid.yaml'
#    path_to_config = '../configs/linear_baseline_configs/covid_configs/linear_delta_per_step_covid_num_events_only.yaml'
#    path_to_config = '../configs/RNN_based_model_configs/covid_configs/RNN_delta_per_step_covid.yaml'
    path_to_config = '../configs/RNN_based_model_configs/covid_configs/learn_fixed_theta_RNN_delta_per_step_covid.yaml'
    params = ParameterParser(path_to_config).parse_params()
    main = DeltaPerStepResultsPlottingMain(params)
    main.main()

