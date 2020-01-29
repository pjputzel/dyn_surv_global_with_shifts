import yaml

def main():
    with open(path_to_config) as f:
        params = yaml.safe_load(f)

    main_type = params['main_type']
    if main_type == 
    elif main_type == 
.
.
.
    else:
        raise ValueError('Main type %s not defined' %str(main_type))    

    Main.main()

if __name__ == '__main__':
    path_to_config = ''
    main(path_to_config)
