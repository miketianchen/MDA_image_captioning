# Author: Fanli Zhou
# Date: 2020-06-09

'''This script 

Usage: src/model/prepare_data.py ROOT_PATH INPUTS...

Arguments:
ROOT_PATH         The root path of the json folder.
INPUTS            The json file name.
'''

import json
from tqdm import tqdm
import pickle
from time import time
import numpy as np
from PIL import Image
from docopt import docopt


START = "startseq"
STOP = "endseq"

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
                f'{root_path}/imgs/{name}/{filename}'
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


def get_embeddings(root_captioning, vocab_size, embedding_dim, wordtoidx):

    embeddings_index = {} 
    
    print('Loading pre-trained Glove embeddings...')

    with open(f'{root_captioning}/glove.6B.200d.txt', 'r', encoding='utf-8') as file:

        for line in tqdm(file):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f'Found {len(embeddings_index)} word vectors.')

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    count = 0

    for word, i in wordtoidx.items():

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            count += 1
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
            
    print(f'{count} out of {vocab_size} words are found in the pre-trained matrix.')            
    print(f'The size of embedding_matrix is {embedding_matrix.shape}')
    return embedding_matrix

if __name__ == "__main__":

    args = docopt(__doc__)
    
    train_paths, train_descriptions, max_length = [], [], 0

    for name in args['INPUTS']:
        paths, descriptions, length =\
        get_img_info(args['ROOT_PATH'], name)
        train_paths.extend(paths)
        train_descriptions.extend(descriptions)
        max_length = max(max_length, length)  

    print(f'{len(train_paths)} images from for training')
    
    # add a start and stop token at the beginning/end
    for v in train_descriptions:
        for d in range(len(v)):
            v[d] = f'{START} {v[d]} {STOP}'

    vocab = get_vocab(train_descriptions, word_count_threshold=10)
    idxtoword, wordtoidx = get_word_dict(vocab)

    max_length += 2   
    vocab_size = len(idxtoword) + 1
    batch_size = 200
    hidden_size = 256
    embedding_dim = 200
    cnn_type = 'vgg16'

    embedding_matrix = get_embeddings(
        args['ROOT_PATH'],
        vocab_size,
        embedding_dim,
        wordtoidx
    ) 

    model_info = {
        'max_length': max_length,
        'idxtoword': idxtoword,
        'wordtoidx': wordtoidx,
        'vocab_size': vocab_size,
    }

    with open( f"{args['ROOT_PATH']}/model/model_info.json", 'w') as f:
        json.dump(model_info, f)
        
    with open(f"{args['ROOT_PATH']}/model/train_paths.pkl", 'wb') as f:
        pickle.dump(train_paths, f)

    with open(f"{args['ROOT_PATH']}/model/train_descriptions.pkl", 'wb') as f:
        pickle.dump(train_descriptions, f)

