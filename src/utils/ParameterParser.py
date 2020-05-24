import yaml

DEFAULT_PARAMS =\
    {
        'metrics_to_track': ['dummy_test_function'],

        'data_input_params': {
            'dataset_name': 'specify in config',
            'cov_time_representation': 'delta',
            'data_loading_params': {
                'paths': 'specify in config'
            },
        },

        'model_params': {
            'model_type': 'basic',
            'covariate_dim': 'specify in config since data-dependent',
            'hidden_dim': 5,
            #'n_layers_rnn': 3,
            'keep_prob': .3,
            'survival_distribution_type': 'Gamma',

        },

        'train_params': {
            'learning_rate': .0001,
            'batch_size': 20,
            'conv_thresh': .0001,
            'n_epoch_print': 1,
            'pretraining': {\
                'pretraining_max_iter': None,
                'regular_training_max_iter': None, 
                'regular_training_lr': None
            },

            'diagnostic_params': {
            },
            
            'loss_params': {
                'distribution_type': 'exponential',
                'avg_per_seq': False
            }
            
        },
        

    }

class ParameterParser:
    def __init__(self, path_to_params):
        self.path_to_params = path_to_params
    
    # only handles a 3 level dictionary
    # could make recursive to handle general case
    # but I think more than 3 levels in the params is confusing
    def parse_params(self):
        self.params = DEFAULT_PARAMS
        with open(self.path_to_params, 'rb') as f:
            new_params = yaml.safe_load(f)
        print('new params', new_params)
        for str_key in new_params:
            if type(new_params[str_key]) == dict:
                for str_key2 in new_params[str_key]:
                    if type(new_params[str_key][str_key2]) == 'dict':
                        self.params[str_key][str_key2].update(new_params[str_key][str_key2])
                    else:
                        self.params[str_key][str_key2] = new_params[str_key][str_key2]
            else:
                self.params[str_key] = new_params[str_key]
        return self.params

                
