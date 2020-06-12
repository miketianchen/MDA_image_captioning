# author: Dora Qian, Fanli Zhou, James Huang, Mike Chen
# date: 2020-06-10

'''This script preprocess the original sydney json file. 
This script takes a raw data folder path and save the files in the desired folder.

Usage: scr/data/preprocess_sydney.py --input_path=<input_path> --output_path=<output_path>

Options:
--input_path=<input_path>  Folder that contains the input files
--output_path=<output_path>  Folder that contains the output files
'''

import json
import os
from docopt import docopt

opt = docopt(__doc__)

def main(input_path, output_path):
    """
    preprocesses the sydney json files with added field of "old_dataset_name" and "split"
    
    Parameters:
    -----------
    input_dir: str
        folder path of the input data
    output_dir: str
        folder path of the output data
        
    Returns:
    -----------
    None, the processed json files will be stored under output directory
    """
    json_data = {}
    sizes = {}

    set_name = ['sydney']

    for name in set_name:
        with open(input_path + '/dataset_' + name + '_modified.json', 'r') as data:
            json_data[name] = json.load(data)
            sizes[name] = len(json_data[name]['images'])
    
    sydney_data = {}

    for name in set_name:
        for single_image in json_data[name]['images']:
            new_filename = name + '_' + single_image['filename'][:-3] + 'jpg'
            if (name == 'sydney'):
                sydney_data[new_filename] = single_image
                sydney_data[new_filename]['old_dataset_name'] = 'dataset_sydney_modified'
            else: 
                print("uh-oh")
    with open(output_path + '/sydney' + '.json', 'w') as file:
        json.dump(sydney_data, file)
        
if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])



