# author: Dora Qian, Fanli Zhou, James Huang, Mike Chen
# date: 2020-06-10

'''This script process the original images convert them into standard format.
This script takes a raw image folder path and save the preprocessed images in the same folder

Usage: scr/data/preprocess_image.py --input_path=<input_path> 

Options:
--input_path=<input_path>  Folder that contains the raw image folders
'''

import os
from PIL import Image
from pathlib import Path
from os import walk
from docopt import docopt

opt = docopt(__doc__)

def main(input_path, output_format = '.jpg', size = (299, 299)):
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
    folders = []
    for (dirpath, dirnames, filenames) in walk(input_path):
        folders.extend(dirnames)
        break
    
    # 'data' directory path (one upstream of directory_path)
    data_directory = str(Path(input_path).parents[0])
    
    for folder in folders:
        dataset_name = folder
        directory_path = input_path  + "/" + folder
        print(directory_path)
        preprocess(dataset_name, directory_path, output_format, size)

def preprocess(dataset_name, directory_path, output_format = '.jpg', size = (299, 299)):
    '''
    Preprocess all the images in a folder. By converting the image
    format to the desired format and resizing the image to the 
    desired size
    
    Parameters:
    ------------
    dataset_name : str
        This name will be appeneded to every image name as well as
        used for the name of directory which will house the images
    directory_path : str
        Absolute path, where the directory containing all the images
        are at. Don't know if relative path works...
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
    processed_path = str(Path(directory_path).parents[1])

    # create folder
    output_path = processed_path + "/preprocessed_" + dataset_name
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(directory_path):   
        # convert everything in the folder that's not the 
        # desired output_format
        if not filename.endswith(output_format):
            # To get rid of the current suffix tag representing
            # current type of image file
            name = filename[:-4]
            name = dataset_name + "_" + name + output_format
            
            # PILLOW library functions 
            im = Image.open(directory_path + "/" + filename).resize(size, Image.ANTIALIAS)
            rgb_im = im.convert('RGB')

            output_path =  processed_path + "/preprocessed_" + dataset_name + "/" + name

            # Pillow Format Type (JPEG, PNG, TIFF)
            if output_format == '.jpg':
                pillow_format = 'JPEG'
            elif output_format == '.png':
                pillow_format = 'PNG'
            elif output_format == '.tif':
                pillow_format = 'TIFF'
            else:
                print("should make a throw statement, and some try catch")

            # Save with Image Quality of 95% 
            rgb_im.save(output_path, pillow_format, quality = 95)

            continue

            # when the image format type is the same, and we just need to resize 
        elif filename.endswith(output_format):
            im = Image.open(directory_path + "/" + filename)
            im = im.resize(size, Image.ANTIALIAS)
            name = dataset_name + "_" + filename
            output_path = processed_path + "/preprocessed_" + dataset_name + "/" + name
            im.save(output_path, quality = 95)
            continue
        else:
            continue 

if __name__ == "__main__":
    main(opt["--input_path"])