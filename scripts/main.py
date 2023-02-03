import argparse
import sys
import traceback
import yaml
from tasks import *
from parse_yaml import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    parser.add_argument('-t', '--task', required=True, type=str, help='Disorder model')
    
    config_filepath = parser.parse_args().config
    task = parser.parse_args().task
    with open(config_filepath, "r") as config_file:
        config = AttrDict(yaml.safe_load(config_file))
        config = list_to_tuples(config)
        
    try:
        os.makedirs(config.setup.root_dir, exist_ok=True)
        for subdir in ['models', 'base', 'figs']:
            os.makedirs(os.path.join(config.setup.root_dir, subdir), exist_ok=True)
    except:
        print(f"Error: cannot create root path.") 
        return -1 

    try:
        globals()[task]
    except Exception as e:
        print(f'{task} not found.')
    globals()[task](config)

    return 0, 'Task successfully executed'

if __name__ == '__main__':
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Task failed.'

    print(status_message)
    exit(retval)
