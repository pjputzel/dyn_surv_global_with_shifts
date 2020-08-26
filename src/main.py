import yaml
import time
from main_types.PreTrainingBasicModelMain import PreTrainingBasicModelMain
from main_types.BasicMain import BasicMain
from main_types.PreTrainingConstantDeltaMain import PreTrainingConstantDeltaMain
from main_types.MultiRunMain import MultiRunMain
from main_types.ModelFreeRankingMain import *
from utils.ParameterParser import ParameterParser
import sys
sys.path.append("..")
from data.make_simple_synth_data import *

def main(path_to_config):
    start_time = time.time()
    params = ParameterParser(path_to_config).parse_params()
    print(params)
    main_type = params['main_type']
    if main_type == 'basic_main':
        main = BasicMain(params)
    elif main_type == 'pretraining_basic_model_main':
        main = PreTrainingBasicModelMain(params)
    elif main_type == 'pretraining_constant_delta_main':
        main = PreTrainingConstantDeltaMain(params)
    elif main_type == 'multi_run':
        main = MultiRunMain(params)
    elif main_type == 'cov_times_ranking_main':
        main = EvaluateCovTimesRankingMain(params)
    elif main_type == 'num_events_ranking_main':
        main = EvaluateNumEventsRankingMain(params)
    else:
        raise ValueError('Main type %s not defined' %str(main_type))    

    main.main()
    print('Total time taken %d' %(time.time() - start_time))

if __name__ == '__main__':
#    path_to_config = '../configs/model_free_configs/synth_configs/cov_times_ranking_synth.yaml'
#    path_to_config = '../configs/model_free_configs/synth_configs/num_event_ranking_synth.yaml'
#    path_to_config = '../configs/model_free_configs/dm_cvd_configs/num_event_ranking_dm_cvd.yaml'

#    path_to_config = '../configs/linear_baseline_configs/synth_configs/linear_theta_per_step_synth.yaml'
    path_to_config = '../configs/linear_baseline_configs/synth_configs/linear_delta_per_step_synth.yaml'

    main(path_to_config)
