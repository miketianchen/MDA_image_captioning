# Author: Fanli Zhou
# Date: 2020-06-09

'''This script 

Usage: scr/models/prepare_data.py --root_path=<root_path> INPUTS ...

Arguments:
INPUTS                   One or more json file names.

Options:
--root_path=<root_path>  The path to the data folder which contains the raw folder.
'''

import json, os, pickle
from tqdm import tqdm
from time import time
import numpy as np
from PIL import Image
from docopt import docopt

START = "startseq"
STOP = "endseq"

np.random.seed(123)

def get_img_info(root_path, name, num=np.inf):
    """
    Returns img paths and captions

    Parameters:
    -----------
    name: str
        the json file name
    num: int (default: np.inf)
        the number of observations to get

    Return:
    --------
    list, dict, int
        img paths, corresponding captions, max length of captions
    """
    img_path = []
    caption = [] 
    max_length = 0
    with open(f'{root_path}/json/{name}.json', 'r') as json_data:
        data = json.load(json_data)
        for filename in data.keys():
            if num is not None and len(caption) == num:
                break
            img_path.append(
                f'{root_path}/{name}/{filename}'
            )
            sen_list = []
            for sentence in data[filename]['sentences']:
                max_length = max(max_length, len(sentence['tokens']))
                sen_list.append(sentence['raw'])

            caption.append(sen_list)    

    return img_path, caption, max_length            

def get_vocab(descriptions, word_count_threshold=10):

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

