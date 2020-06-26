# author: Dora Qian, Fanli Zhou, James Huang, Mike Chen
# date: 2020-06-10

'''This script preprocess the original json file.

This script takes the root path to the raw data and image folders to process.

Usage: scr/data/preprocess_json.py --root_path=<root_path> INPUTS ...

Arguments:
INPUTS                     Datasets to process (e.g. rsicd, ucm, sydney, ...).

Options:
--root_path=<root_path>    The path to the data folder which contains the raw folder.

Example:

python scr/data/preprocess_json.py --root_path=data ucm rsicd sydney

Preprocess `data/raw/dataset_ucm_modified.json`, `data/raw/dataset_rsicd_modified.json`,
and `data/raw/dataset_sydney_modified.json` to standardize json format, and save outputs as
`data/json/ucm.json`, `data/json/rsicd.json`, and `data/json/sydney.json`

'''

import json
import os
from docopt import docopt
from collections import defaultdict

args = docopt(__doc__)

def main(root_path, inputs):
    """
    preprocesses the json files with added field of "old_dataset_name" 
    
    Parameters:
    -----------
    root_path: str
        The root path of the raw folder
    inputs: str
        Datasets to process (e.g. rsicd, ucm, sydney)
    train: str
        Prepare json files for training or not.
        
    Returns:
    -----------
    None, the processed json files will be stored under output directory
    """

    data_info = defaultdict(dict)
    # read in json files from all datasets
    for name in inputs:
        
        print(f'Processing file information for the {name} dataset...')
        
        try:
            with open(f'{root_path}/raw/dataset_{name}_modified.json', 'r') as data:
                json_data = json.load(data)
            for single_image in json_data['images']:
                new_filename = f"{name}_{os.path.splitext(single_image['filename'])[0]}.jpg"
                data_info[name][new_filename] = single_image
                data_info[name][new_filename]['old_dataset_name'] = f'dataset_{name}_modified'
        except:
            with open(f'{root_path}/raw/{name}.json', 'r') as data:
                json_data = json.load(data)
            for filename, value in json_data.items():
                new_filename = f"{name}_{os.path.splitext(filename)[0]}.jpg"
                data_info[name][new_filename] = value
                data_info[name][new_filename]['filename'] = filename
                data_info[name][new_filename]['old_dataset_name'] = name

    output_path = f'{root_path}/json'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True) 

    for name in inputs:
        with open(f'{output_path}/{name}.json', 'w') as file:
            json.dump(data_info[name], file)


if __name__ == "__main__":
    main(args["--root_path"], args["INPUTS"])
    