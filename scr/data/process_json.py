# author: Dora Qian, Fanli Zhou, James Huang, Mike Chen
# date: 2020-06-10

'''This script process the original json file and correct problematic captions. 
This script takes a raw data folder path and save the files in the desired folder.

Usage: scr/data/process_json.py --input_path=<input_path> --output_path=<output_path>

Options:
--input_path=<input_path>  Folder that contains the input files
--output_path=<output_path>  Folder that contains the output files
'''

import json
import pandas as pd
import os
import random
from docopt import docopt
from sklearn.model_selection import train_test_split

opt = docopt(__doc__)

def main(input_path, output_path):
    """
    Combines the rscid and ucm datasets and splits into 80%/20% trainvalid/test and train/valid set,
    preprocesses the json files with added field of "old_dataset_name" and "split"
    
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

    set_name = ['rsicd', 'ucm']

    # read in json files from all datasets
    for name in set_name:
        with open(input_path + '/dataset_' + name + '_modified.json', 'r') as data:
            json_data[name] = json.load(data)
            sizes[name] = len(json_data[name]['images'])

    ucm_data = {}
    sydney_data = {}
    rsicd_data = {}

    for name in set_name:
        for single_image in json_data[name]['images']:
            new_filename = name + '_' + single_image['filename'][:-3] + 'jpg'
            if (name == 'rsicd'):
                rsicd_data[new_filename] = single_image
                rsicd_data[new_filename]['old_dataset_name'] = 'dataset_rsicd_modified'
            elif (name == 'ucm'):
                ucm_data[new_filename] = single_image
                ucm_data[new_filename]['old_dataset_name'] = 'dataset_ucm_modified'
            elif (name == 'sydney'):
                sydney_data[new_filename] = single_image
                sydney_data[new_filename]['old_dataset_name'] = 'dataset_sydney_modified'
            else: 
                print("New folder is found:", name) 

    combined_data = {**rsicd_data, **ucm_data, **sydney_data}
    combined_df = pd.DataFrame(combined_data).T

    # split the data by 80%/20% 
    train_valid, test = train_test_split(combined_df, test_size=0.2, random_state=123)
    train, valid = train_test_split(train_valid, test_size=0.2, random_state=123)

    train_dict = train.to_dict(orient='index')
    test_dict = test.to_dict(orient='index')
    valid_dict = valid.to_dict(orient='index')

    # In each of the images add the current set name key-value pair
    new_set_name = ['train', 'test', 'valid']
    for name in new_set_name:
        if (name == 'train'):
            for key, value in train_dict.items():
                value['split'] = name
        elif (name == 'test'):
            for key, value in test_dict.items():
                value['split'] = name
        elif (name == 'valid'):
            for key, value in valid_dict.items():
                value['split'] = name
        else:
            print("uh-oh")
            
    # correct the captions in train json
    train_dict = corret_captions(train_dict)

    imgs_names = [(train_dict, 'train', train), 
                  (valid_dict, 'valid', valid), 
                  (test_dict, 'test', test)]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # save the processed json
    for imgs, name, _ in imgs_names:
        with open(output_path + '/' + name + '.json', 'w') as file:
            json.dump(imgs, file)
            
def corret_captions(train_data):
    """
    docstring
    """
    random.seed(18)
    data = train_data
    
    # create problematic caption dictionary
    count_err = 0
    err_cap_dict = {}
    for file in data.keys():
        for caption in data[file]['sentences']:
            if len(caption['tokens']) < 2:
                count_err += 1
                if file in err_cap_dict.keys():
                    err_cap_dict[file].append(caption['sentid'])
                else:
                    err_cap_dict[file] = [caption['sentid']]
                    
    # print("The number of problematic caption found is", count_err)
    # print(err_cap_dict)
    
    # replace the problematic caption with random selected caption from the same image
    for key, val in err_cap_dict.items():
        caption_list = [sentence['sentid'] for sentence in data[key]['sentences'] if sentence['sentid'] not in val]
        for sentence in data[key]['sentences']:
            if sentence['sentid'] in val:

                sel_caption = random.choice(caption_list)
                sel_tokens = [sentence['tokens'] for sentence in data[key]['sentences'] if sentence['sentid'] == sel_caption][0]
                sel_raw = [sentence['raw'] for sentence in data[key]['sentences'] if sentence['sentid'] == sel_caption][0]

                sentence['tokens'] = sel_tokens
                sentence['raw'] = sel_raw
                
    return data
        
def test(output_path):
    """
    check whether the outputs has been generated successfully 
    
    Parameters:
    -----------
    output_dir: str
        folder path of the output data
    
    Returns:
    -------
        passes if the test is successful, otherwise returns error messages
    """
    assert os.path.isfile(output_path +  '/train.json'), "Train json is not generated."
    assert os.path.isfile(output_path +  '/valid.json'), "Valid json is not generated."
    assert os.path.isfile(output_path +  '/test.json'), "Test json is not generated."

if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])
    test(opt["--output_path"])