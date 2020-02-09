import yaml
from  main_types.BasicMain import BasicMain
from  utils.ParameterParser import ParameterParser

def main(path_to_config):
    params = ParameterParser(path_to_config).parse_params()
    print(params)
    main_type = params['main_type']
    if main_type == 'basic_main':
        main = BasicMain(params)
    else:
        raise ValueError('Main type %s not defined' %str(main_type))    

    main.main()

if __name__ == '__main__':
    path_to_config = '../configs/basic_main.yaml'
    main(path_to_config)
