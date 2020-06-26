# author: Dora Qian, Fanli Zhou, James Huang, Mike Chen
# date: 2020-06-10

'''This script splits the combined datasets into train, test and 
valid datasets and correct problematic captions in the train dataset.

Usage: scr/data/split_data.py --root_path=<root_path> INPUTS ... 

Arguments:
INPUTS                       Datasets to process (e.g. rsicd, ucm, sydney).

Options:
--root_path=<root_path>      The path to the data folder which contains the raw folder.

Example:

python scr/data/split_data.py --root_path=data ucm rsicd

Combine `data/json/ucm.json` and `data/json/rsicd.json`, split the combined dataset into
train, valid, test datasets, and save results in `data/json/train.json`, `data/json/valid.json`, 
and `data/json/test.json`
'''

import json
import pandas as pd
import random
import os
from docopt import docopt
from sklearn.model_selection import train_test_split

def corret_captions(train_data):
    """
    Correct the problematic captions based on eda notebooks under notebook folders. Replace any captions less than 2 tokens with a random selected captions of the same image.
    
    Parameters:
    ----------
    train_data : dict
        the dictionary contains train captions
    
    Returns:
    -----------
    dict, the corrected version of train dict
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
        
def tests(root_path):
    """
    check whether the outputs has been generated successfully 
    
    Parameters:
    -----------
    root_path: str
        the path to the data folder which contains the raw folder
    
    Returns:
    -------
        passes if the test is successful, otherwise returns error messages
    """
    assert os.path.isfile(f'{root_path}/json/train.json'), "Train json is not generated."
    assert os.path.isfile(f'{root_path}/json/valid.json'), "Valid json is not generated."
    assert os.path.isfile(f'{root_path}/json/test.json'), "Test json is not generated."
    
if __name__ == "__main__":
    args = docopt(__doc__)
    root_path = args["--root_path"]
    inputs = args["INPUTS"]
    
    combined_data = {}
    print(f'Combining {inputs} datasets...') 
    
    # read in json files from all datasets
    for name in inputs:
             
        with open(f'{root_path}/json/{name}.json', 'r') as data:
            json_data = json.load(data)
        
        combined_data.update(json_data)
        
    combined_df = pd.DataFrame(combined_data).T

    print(f'Splitting data into train, valid and test sets...')

    # split the data by 80%/20% 
    train_valid, test = train_test_split(combined_df, test_size=0.2, random_state=123)
    train, valid = train_test_split(train_valid, test_size=0.2, random_state=123)

    train_dict = train.to_dict(orient='index')
    test_dict = test.to_dict(orient='index')
    valid_dict = valid.to_dict(orient='index')

    # In each of the images add the current set name key-value pair

    for name in ['train', 'test', 'valid']:
        if (name == 'train'):
            for key, value in train_dict.items():
                value['split'] = name
        elif (name == 'test'):
            for key, value in test_dict.items():
                value['split'] = name
        else:
            for key, value in valid_dict.items():
                value['split'] = name

    # correct the captions in train json
    train_dict = corret_captions(train_dict)

    imgs_names = [(train_dict, 'train', train), 
                  (valid_dict, 'valid', valid), 
                  (test_dict, 'test', test)]

    # save the processed json
    for imgs, name, _ in imgs_names:
        with open(f'{root_path}/json/{name}.json', 'w') as file:
            json.dump(imgs, file)
            
    tests(root_path)