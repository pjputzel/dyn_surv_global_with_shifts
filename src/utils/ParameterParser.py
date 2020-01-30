import yaml

DEFAULT_PARAMS =\
    {
        'metrics_to_track': ['dummy_test_function'],

        'data_input_params': {
            'dataset_name': 'specify in config',
            'data_loading_params': {
                'paths': 'specify in config'
            },
        },

        'model_params': {
            'covariate_dim': 'specify in config since data-dependent',
            'hidden_dim': 5,
            'n_layers_rnn': 3,
            'dropout': .3,
            'survival_distribution_type': 'GGD',

        },

        'train_params': {
            'learning_rate': .0001,
            'batch_size': 20,
            'conv_thresh': .0001,
         

        },

    }

class ParameterParser:
    def __init__(self, path_to_params):
        self.path_to_params = path_to_params
    
    def parse_params(self):
        self.params = DEFAULT_PARAMS
        with open(self.path_to_params, 'rb') as f:
            new_params = yaml.safe_load(f)
        print('new params', new_params)
        for str_key in new_params:
            if type(new_params[str_key]) == dict:
                if str_key == 'transform_params':
                    self.update_transform_params(new_params[str_key])
                else:
                    self.params[str_key].update(new_params[str_key])
            else:
                self.params[str_key] = new_params[str_key]
        return self.params

    def update_transform_params(self, new_params):
        for key in new_params:
            if type(new_params[key]) == dict:
                self.params['transform_params'][key].update(new_params[key])
            else:
                self.params['transform_params'][key] = new_params[key]
                
