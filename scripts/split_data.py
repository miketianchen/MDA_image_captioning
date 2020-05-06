import json
import numpy as np
from collections import defaultdict
from functools import reduce
from sklearn.model_selection import train_test_split


def split_data(seqs, label):
    """
    Extracts image information for the training, validation,
    or test dataset.
    
    Parameters:
    ------------
    seqs: numpy.ndarray
        the indexes for the dataset
    label: str
        the name of the dataset
        
    Return:
    --------
    dict
        a dict with image infromation with the following 
        structure: 
        {dataset name (one of 'rsicd', 'ucm' and 'sydney'): 
            {
                filename (in the format of *.tif or *.jpeg): 
                dict of the img info
            }
        
        }
        
    """
    
    
    def aggerate(x, ind, name):
        """
        Aggerates image information.

        Parameters:
        ------------
        x: dict
            the dict to store image information
        ind: int
            the index of the image in the dataset
        name: str
            the name of the dataset 
            (one of 'rsicd', 'ucm' and 'sydney')
        Return:
        --------
        dict
            x
        """
        x[json_data[name]['images'][ind]['filename']] = json_data[name]['images'][ind]
        return x
    
    print()
    print(f'Preparing the {label} dataset:')

    imgs = {}
    
    imgs['rsicd'] = reduce(lambda x, y: aggerate(x, y, 'rsicd'),
                           seqs[seqs < sizes['rsicd']], {})
    
    imgs['ucm'] = reduce(lambda x, y: aggerate(x, y - sizes['rsicd'], 'ucm'),
                         seqs[(seqs >= sizes['rsicd']) & (seqs < sizes['rsicd'] + sizes['ucm'])], {})
    
    imgs['sydney'] = reduce(lambda x, y: aggerate(x, y - sizes['rsicd'] - sizes['ucm'], 'sydney'),
                            seqs[sizes['rsicd'] + sizes['ucm'] <= seqs], {})

    print(f"{len(imgs['rsicd'])} images from the RSICD dataset")
    print(f"{len(imgs['ucm'])} images from the UCM dataset")
    print(f"{len(imgs['sydney'])} images from the Sydney dataset")
    print(f"{len(imgs['rsicd']) + len(imgs['ucm']) + len(imgs['sydney'])} images in total")

    return imgs

# main
json_data = {}
sizes = {}
set_name = ['rsicd', 'ucm', 'sydney']

# read in json files from all three datasets
for name in set_name:
    with open('../data/raw_data/dataset_' + name + '_modified.json', 'r') as data:
        json_data[name] = json.load(data)
        sizes[name] = len(json_data[name]['images'])
        print(f'There are {sizes[name]} images in the {name} dataset.')

# create splits based on a sequence from 0 to 13633
# an image from the RSCID dataset has a index in [0, 10920]
# an image from the UCM dataset has a index in [10921, 13020]
# an image from the Sydney dataset has a index in [13021, 13633]
train_valid, test = train_test_split(np.arange(sum(list(sizes.values()))), test_size=0.2, random_state=123)
train, valid = train_test_split(train_valid, test_size=0.2, random_state=123)

train_imgs = split_data(train, 'training')
valid_imgs = split_data(valid, 'validation')
test_imgs = split_data(test, 'test')

imgs_names = [(train_imgs, 'train', train), 
              (valid_imgs, 'valid', valid), 
              (test_imgs, 'test', test)]

# test the split_data function
for imgs, name, seq in imgs_names:
    
    for key in set_name:
        assert len(set((imgs[key].keys()))) == len(list(imgs[key].keys())),\
        f'There is duplicated image from the {key} dataset in the {name} dataset.' 
        
    assert len(seq) == sum([len(list(imgs[key].keys())) for key in set_name]),\
        f'The number of the {name} images does not match the size of the {name} dataset.'

for imgs, name, _ in imgs_names:
    with open('../data/clean_data/' + name + '.json', 'w') as file:
        json.dump(imgs, file)