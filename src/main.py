import yaml
import click
import time
#from main_types.PreTrainingBasicModelMain import PreTrainingBasicModelMain
from main_types.BasicMain import BasicMain
from main_types.EvalSavedResultsMain import EvalSavedResultsMain
from main_types.PreTrainingConstantDeltaMain import PreTrainingConstantDeltaMain
from main_types.LearnFixedThetaBasicMain import LearnFixedThetaBasicMain
from main_types.MultiRunMain import MultiRunMain
from main_types.ModelFreeRankingMain import *
from main_types.ValidateHiddenDimMain import ValidateHiddenDimMain
from main_types.ValidateHiddenDimMain import ParallelValidateHiddenDimMain
from main_types.SaveLearnedStandardRisksMain import SaveLearnedStandardRisksMain
from main_types.LandmarkedCoxMain import LandmarkedCoxMain
#from main_types import *
from utils.ParameterParser import ParameterParser
#from data_handling.COVIDSevereOutcomePreprocessor import *
import sys
#import os
#sys.path.append(os.getcwd())
sys.path.append("..")
from data.make_simple_synth_data import *
sys.path.append('../data/COVID-19/')
# for use with line profiler which doesn't add this to the path for some reason
#sys.path.append('/home/pj/Documents/Dynamic SA/DDGGD/DDGGD/src')
import preprocess_data
from preprocess_data import COVID19SevereOutcomePreprocessor
#from preprocess_data_old import COVID19_Preprocessor
#from data_handling.COVIDSevereOutcomePreprocessor import COVID19SevereOutcomePreprocessor

@click.command()
@click.option('--config', help='path to configuration file in yaml format')
def main(config):
    start_time = time.time()
    params = ParameterParser(config).parse_params()
    print(params)
    main_type = params['main_type']
    if main_type == 'basic_main':
        main = BasicMain(params)
#    elif main_type == 'pretraining_basic_model_main':
#        main = PreTrainingBasicModelMain(params)
    elif main_type == 'learn_fixed_theta_basic_main':
        main = LearnFixedThetaBasicMain(params)
    elif main_type == 'pretraining_constant_delta_main':
        main = PreTrainingConstantDeltaMain(params)
    elif main_type == 'multi_run':
        main = MultiRunMain(params)
    elif main_type == 'cov_times_ranking_main':
        main = EvaluateCovTimesRankingMain(params)
    elif main_type == 'num_events_ranking_main':
        main = EvaluateNumEventsRankingMain(params)
    elif main_type == 'framingham_ranking_main':
        main = EvaluateFraminghamRankingMain(params)
    elif main_type == 'validate_RNN_hidden_dim_main':
        main = ValidateHiddenDimMain(params)
    elif main_type == 'parallel_validate_RNN_hidden_dim_main':
        main = ParallelValidateHiddenDimMain(params)
    elif main_type == 'eval_saved_results_main':
        main = EvalSavedResultsMain(params)
    elif main_type == 'save_learned_risks_standard_c_index_main':
        main = SaveLearnedStandardRisksMain(params)
    elif main_type == 'landmarked_cox_main':
        main = LandmarkedCoxMain(params)
    else:
        raise ValueError('Main type %s not defined' %str(main_type))    

    main.main()
    print('Total time taken %d' %(time.time() - start_time))

if __name__ == '__main__':
    main()

