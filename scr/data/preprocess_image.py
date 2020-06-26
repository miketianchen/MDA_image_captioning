# author: Dora Qian, Fanli Zhou, James Huang, Mike Chen
# date: 2020-06-10

'''This script process the original images convert them into standard format.
This script takes a raw image folder path and save the preprocessed images in the same folder

Usage: scr/data/preprocess_image.py --root_path=<root_path> INPUTS ... [--train=<train>]

Arguments:
INPUTS                     Path to the folder that contains the raw images to process under the root_path (e.g. raw).

Options:
--root_path=<root_path>    The path to the data folder which contains the raw folder.
--train=<train>            Sort the images if the images are for training [default: False].

Examples:

Case 1:

python scr/data/preprocess_image.py --root_path=data raw/ucm raw/rsicd --train=True

Preprocess the images in `data/raw/ucm` and `data/raw/rsicd`, and sort the
preprocessed images into `data/train` `data/valid` and `data/test` folders
based on `data/json/train.json` `data/json/valid.json` and `data/json/test.json`. 

Case 2:

python scr/data/preprocess_image.py --root_path=data raw/sydney

Preprocess the images in `data/raw/sydney`, and save the preprcessed images in
`data/preprocessed_sydney`.
'''

import os, json
from PIL import Image
from pathlib import Path
from os import walk
from docopt import docopt

args = docopt(__doc__)

def main(root_path, inputs, train):
    """
    Preprocess all the image folders in the input directory in a folder.
    
    Parameters:
    ------------
    input_path : str
        the path contains all the raw image folders
    output_format : str
        output format you want example (.jpg, .png, .tif) 
        INCLUDE THE PERIOD before the type
    size : tuple of int (WIDTH, HEIGHT)
        size of the image to be converted 
    
    Return:
    --------
    None, it will create a folder in the same directory
    as the image folder you wanted to preprocess, with all the pre-
    processed images inside
    """
 
    for folder in inputs:
        dataset_name = folder.split('/')[-1]
        directory_path = f'{root_path}/{folder}'
        print(f'Preprocessing images in {directory_path}...')
        preprocess(dataset_name, root_path, directory_path)

    if train == 'True':
        sort_image(root_path)
    
def preprocess(dataset_name, root_path, directory_path, output_format = '.jpg', size = (299, 299)):
    '''
    Preprocess all the images in a folder. By converting the image
    format to the desired format and resizing the image to the 
    desired size
    
    Parameters:
    ------------
    dataset_name : str
        This name will be appeneded to every image name as well as
        used for the name of directory which will house the images
    root_path : str
        Relative path to the data folder
    directory_path : str
        Relative path to the image folder under the root_path
    output_format : str
        output format you want example (.jpg, .png, .tif) 
        INCLUDE THE PERIOD before the type
    size : tuple of int (WIDTH, HEIGHT)
        size of the image to be converted 
    
    Return:
    --------
    None, it will create a folder in the same directory
    as the image folder you wanted to preprocess, with all the pre-
    processed images inside
    '''

    # create folder
    folder_path = f'{root_path}/preprocessed_{dataset_name}'
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    for filename in os.listdir(directory_path): 
        
        try:
            im = Image.open(f'{directory_path}/{filename}').resize(size, Image.ANTIALIAS)
            name, extension = os.path.splitext(filename)
            name = f'{dataset_name}_{name}{output_format}'
            output_path = f'{folder_path}/{name}'

            # convert everything in the folder that's not the 
            # desired output_format
            if extension != output_format:

                # PILLOW library functions 
                rgb_im = im.convert('RGB')

                # Pillow Format Type (JPEG, PNG, TIFF)
                if output_format == '.jpg':
                    pillow_format = 'JPEG'
                elif output_format == '.png':
                    pillow_format = 'PNG'
                elif output_format == '.tif':
                    pillow_format = 'TIFF'
                else:
                    raise("The output format should be one of .jpg, .png or .tif")

                # Save with Image Quality of 95% 
                rgb_im.save(output_path, pillow_format, quality = 95)

            else:
                im.save(output_path, quality = 95)
        except:
            print(f'Found unexpect file {filename} in {directory_path}.')

def sort_image(root_path):
    
    sizes = {}
    set_names = ['test', 'train', 'valid']
    dataset_list = set()

    for set_name in set_names:
        
        print(f'Moving images to the {set_name} folder...')
        
        with open(f'{root_path}/json/{set_name}.json', 'r') as data:
            json_data = json.load(data)
            
        # Make new directory to house the preprocessed images 
        output_dir = f'{root_path}/{set_name}'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for filename in json_data.keys():
            destination_path = f'{output_dir}/{filename}'

            # Split the filename with `_` to find the dataset it originates from 
            dataset = filename.split('_')[0]
            dataset_list.add(dataset)

            origin_path = f'{root_path}/preprocessed_{dataset}/{filename}'        
            os.rename(origin_path, destination_path)
            
    # remove folder if empty
    for set_name in dataset_list:
        
        folder_path = f'{root_path}/preprocessed_{set_name}'
        if len(os.listdir(folder_path)) == 0:
            os.rmdir(folder_path)
    
if __name__ == "__main__":
    main(args["--root_path"], args["INPUTS"], args["--train"])