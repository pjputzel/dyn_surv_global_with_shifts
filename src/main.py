import yaml
import main_types.BasicMain
import utils.ParameterParser

def main():
    params = ParameterParser(path_to_config).parse_params()

    main_type = params['main_type']
    if main_type == 'basic_main':
        main = BasicMain(params)
    else:
        raise ValueError('Main type %s not defined' %str(main_type))    

    main.main()

if __name__ == '__main__':
    path_to_config = '../configs/basic_main.yaml'
    main(path_to_config)
