#!/usr/bin/env python
# coding: utf-8

# In[23]:


import os
from PIL import Image
from pathlib import Path


# In[24]:


from os import walk
PATH = "data/raw" 


# In[25]:


PATH


# In[26]:


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
    nothing explict, but it'll create a folder in the same directory
    as the image folder you wanted to preprocess, with all the pre-
    processed images inside
    '''
    ## Get the base path to the data folder which contains everything..
    #base_path = str(Path(SYDNEY_PATH).parents[1])
    
    base_path = str(Path(directory_path).parents[1])
    
    # The path for the processed directory, houses all the processed images
    processed_path = base_path + "/processed/"
    
    ###!!!!!! MAKE SURE YOU DON'T HAVE `preprocessed_rsicd`, `preprocessed_ucm` and `preprocessed_sydney` 
    ###!!!!!! in the ./preprocessed/ directory. This function will make them for you. !!!!!!!!!!!!!!!!!!!
    # Make new directory to house the preprocessed images 
    os.mkdir(processed_path + "preprocessed_" + dataset_name)

    for filename in os.listdir(directory_path):
        # debugging purposes 
        #print(filename)
        

        
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

            output_path =  processed_path + "preprocessed_" + dataset_name + "/" + name

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
            output_path = processed_path + "preprocessed_" + dataset_name + "/" + name
            im.save(output_path, quality = 95)
            continue
        else:
            continue 


# In[27]:


def process_all_folders_in_directory(raw_dataset_path, output_format = '.jpg', size = (299, 299)):
    folders = []
    for (dirpath, dirnames, filenames) in walk(raw_dataset_path):
        folders.extend(dirnames)
        break
    
    # 'data' directory path (one upstream of directory_path)
    data_directory = str(Path(raw_dataset_path).parents[0])
    
    for folder in folders:
        dataset_name = folder.split('_')[0]
        directory_path = raw_dataset_path + "/" + folder
        preprocess(dataset_name, directory_path, output_format, size)


# In[28]:


process_all_folders_in_directory(PATH)


# In[ ]:




