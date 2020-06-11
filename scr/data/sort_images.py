# author: Dora Qian, Fanli Zhou, James Huang, Mike Chen
# date: 2020-06-10

'''This script places the preprocessed images into train/valid/test folders according to the splitting json files. This script takes paths of json folder, preprocessed image folders and save the preprocessed images in the correct folder.

Usage: scr/data/sort_images.py --json_path=<json_path> --img_path=<img_path> --output_path=<output_path>

Options:
--json_path=<json_path> dddrs
--img_path=<img_path> dddrs
--output_path=<output_path> dddrs
'''

import os
import json
from docopt import docopt

opt = docopt(__doc__)

def main(json_path, img_path, output_path):
    json_data = {}
    sizes = {}
    set_names = ['test', 'train', 'valid']
    dataset_list = []

    for name in set_names:
        with open(json_path + '/' + name + '.json', 'r') as data:
            json_data[name] = json.load(data)
    
    for set_name in set_names:
    # Make new directory to house the preprocessed images 
        output_dir = output_path + '/' + set_name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for filename in json_data[set_name].keys():
            destination_path = output_dir + '/' + filename

            # Split the filename with `_` to find the dataset it originates from 
            dataset = filename.split('_')[0]
            if dataset not in dataset_list:
                dataset_list.append(dataset)

            origin_path = img_path + '/' + 'preprocessed_'+ dataset  + '/' + filename        

            os.rename(origin_path, destination_path)
            
    # remove folder if empty
    for data_set in dataset_list:
        folder_path = img_path + '/' + 'preprocessed_'+ data_set
        if len(os.listdir(folder_path)) == 0:
            os.rmdir(folder_path)
    
if __name__ == "__main__":
    main(opt["--json_path"], opt["--img_path"], opt["--output_path"])