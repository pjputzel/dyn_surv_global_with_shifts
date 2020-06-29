import yaml
import time
from main_types.PreTrainingBasicModelMain import PreTrainingBasicModelMain
from main_types.BasicMain import BasicMain
from main_types.PreTrainingConstantDeltaMain import PreTrainingConstantDeltaMain
from utils.ParameterParser import ParameterParser

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
    else:
        raise ValueError('Main type %s not defined' %str(main_type))    

    main.main()
    print('Total time taken %d' %(time.time() - start_time))

if __name__ == '__main__':
 #   path_to_config = '../configs/basic_main.yaml'
 #   path_to_config = '../configs/theta_per_step_main.yaml'
    path_to_config = '../configs/linear_constant_delta.yaml'
 #   path_to_config = '../configs/delta_per_step.yaml'
    main(path_to_config)
