# Author: Fanli Zhou
# Date: 2020-06-09

'''This script prepares data for training.

Usage: scr/models/prepare_data.py --root_path=<root_path> INPUTS ...

Arguments:
INPUTS                   One or more json file names.

Options:
--root_path=<root_path>  The path to the data folder which contains the raw folder.

Example:
python scr/models/prepare_data.py --root_path=data train

Extract image paths info and related captions from `data/json/train.json`,
and save train image paths in `data/results/train_paths.pkl` and train captions 
in `data/results/train_descriptions.pkl`.
'''

import os, json, pickle
from tqdm import tqdm
from time import time
import numpy as np
from PIL import Image
from docopt import docopt

START = "startseq"
STOP = "endseq"

np.random.seed(123)

def get_img_info(root_path, name):
    """
    Returns img paths and captions

    Parameters:
    -----------
    root_path: str
        the path to the data folder which contains the raw folder
    name: str
        the json file name

    Return:
    --------
    list, dict, int
        img paths, corresponding captions, max length of captions
    """
    img_path = []
    caption = [] 
    max_length = 0
    img_dir = f'{root_path}/{name}'
    if not os.path.exists(img_dir):
        img_dir = f'{root_path}/preprocessed_{name}'
        
    with open(f'{root_path}/json/{name}.json', 'r') as json_data:
        data = json.load(json_data)
        for filename in data.keys():
                
            img_path.append(
                f'{img_dir}/{filename}'
            )
                
            sen_list = []
            for sentence in data[filename]['sentences']:
                max_length = max(max_length, len(sentence['tokens']))
                sen_list.append(sentence['raw'])

            caption.append(sen_list)    

    return img_path, caption, max_length            

def get_vocab(descriptions, word_count_threshold=10):
    """
    Get the vocabulary

    Parameters:
    -----------
    descriptions: list
        a list of captions
    word_count_threshold: int (default: 10)
        the threshold to keep words in the vocabulary

    Return:
    --------
    list
        the vocabulary
    """

    captions = []
    for val in descriptions:
        for cap in val:
            captions.append(cap)
    print(f'{len(captions)} captions for training')
    
    word_counts = {}
    nsents = 0
    for sent in captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print(f'Removed uncommon words (< {word_count_threshold}) and',
          f'reduced the number of words from {len(word_counts)} to {len(vocab)}')
    return vocab

def get_word_dict(vocab):
    """
    Get the dict to get word and the dict to get word index

    Parameters:
    -----------
    vocab: list
        the vocabulary

    Return:
    --------
    dict, dict
        the dict to get word, the dict to get word index
    """

    idxtoword = {}
    wordtoidx = {}

    ix = 1
    for w in vocab:
        wordtoidx[w] = ix
        idxtoword[ix] = w
        ix += 1

    return idxtoword, wordtoidx

if __name__ == "__main__":

    args = docopt(__doc__)
    root_path = args['--root_path']
    train_paths, train_descriptions, max_length = [], [], 0

    for name in args['INPUTS']:
        paths, descriptions, length =\
        get_img_info(root_path, name)
        train_paths.extend(paths)
        train_descriptions.extend(descriptions)
        max_length = max(max_length, length)  

    print(f'{len(train_paths)} images for training')
    
    # add a start and stop token at the beginning/end
    for v in train_descriptions:
        for d in range(len(v)):
            v[d] = f'{START} {v[d]} {STOP}'

    vocab = get_vocab(train_descriptions, word_count_threshold=10)
    idxtoword, wordtoidx = get_word_dict(vocab)

    max_length += 2   
    vocab_size = len(idxtoword) + 1

    model_info = {
        'max_length': max_length,
        'vocab_size': vocab_size,
        'idxtoword': idxtoword,
        'wordtoidx': wordtoidx
    }
    
    if not os.path.exists(f"{root_path}/results"):
        os.makedirs(f"{root_path}/results", exist_ok=True)
        
    with open( f"{root_path}/results/model_info.json", 'w') as f:
        json.dump(model_info, f)
        
    with open(f"{root_path}/results/train_paths.pkl", 'wb') as f:
        pickle.dump(train_paths, f)

    with open(f"{root_path}/results/train_descriptions.pkl", 'wb') as f:
        pickle.dump(train_descriptions, f)

    assert os.path.isfile(f'{root_path}/results/model_info.json'),\
        "Model information is not saved."
    assert os.path.isfile(f'{root_path}/results/train_paths.pkl'),\
        "Train paths are not saved."
    assert os.path.isfile(f'{root_path}/results/train_descriptions.pkl'),\
        "Train descriptions are not saved."
