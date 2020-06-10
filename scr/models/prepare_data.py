# prepare_data.py
#
# Author: Fanli Zhou
#
# Date: 2020-06-09
#
# This script defines get_img_info, get_vocab, get_word_dict,
# get_embeddings, SampleDataset, my_collate, encode_image,
# extract_img_features, hms_string
# 

import json
from tqdm import tqdm
import pickle
from time import time
import numpy as np
from PIL import Image
from itertools import chain

import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset



def get_img_info(root_captioning, name, num=np.inf):
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
    with open(f'{root_captioning}/json/{name}.json', 'r') as json_data:
        data = json.load(json_data)
        for filename in data.keys():
            if num is not None and len(caption) == num:
                break
            img_path.append(
                f'{root_captioning}/{name}/{filename}'
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
    print(f'There are {len(captions)} captions')
    
    word_counts = {}
    nsents = 0
    for sent in captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))
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

class SampleDataset(Dataset):
    def __init__(
        self,
        descriptions,
        imgs,
        wordtoidx,
        max_length
        ):
        """
        Initializes a SampleDataset

        Parameters:
        -----------
        descriptions: list
            a list of captions
        imgs: numpy.ndarray
            the image features
        wordtoidx: dict
            the dict to get word index
        max_length: int
            all captions will be padded to this size
        """        
        self.imgs = imgs
        self.descriptions = descriptions
        self.wordtoidx = wordtoidx
        self.max_length = max_length

    def __len__(self):
        """
        Returns the batch size

        Return:
        --------
        int
            the batch size
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Prepare data for each image

        Parameters:
        -----------
        idx: int
          the index of the image to process

        Return:
        --------
        list, list, list
            [5 x image feature matrix],
            [five padded captions for this image]
            [the length of each caption]
        """

        img = self.imgs[idx]
        img_features, captions = [], []
        for desc in self.descriptions[idx]:
            # convert each word into a list of sequences.
            seq = [wordtoidx[word] for word in desc.split(' ')
                  if word in self.wordtoidx]
            # pad the sequence with 0 on the right side
            in_seq = np.pad(
                seq, 
                (0, max_length - len(seq)),
                mode='constant',
                constant_values=(0, 0)
                )

            img_features.append(img)
            captions.append(in_seq)
    
        return (img_features, captions)

def my_collate(batch):
    """
    Processes the batch to return from the dataloader

    Parameters:
    -----------
    batch: tuple
      a batch from the Dataset

    Return:
    --------
    list
        [image feature matrix, captions, the length of each caption]
    """  

    img_features = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    img_features = torch.FloatTensor(list(chain(*img_features)))
    captions = torch.LongTensor(list(chain(*captions)))

    return [img_features, captions]

def encode_image(model, img_path, device):
    """
    Process the images to extract features

    Parameters:
    -----------
    model: CNNModel
      a CNNModel instance
    img_path: str
        the path of the image
 
    Return:
    --------
    torch.Tensor
        the extracted feature matrix from CNNModel
    """  

    img = Image.open(img_path)

    # Perform preprocessing needed by pre-trained models
    preprocessor = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = preprocessor(img)
    # Expand to 2D array
    img = img.view(1, *img.shape)
    # Call model to extract the smaller feature set for the image.
    x = model(img.to(device), False) 
    # Shape to correct form to be accepted by LSTM captioning network.
    x = np.squeeze(x)
    return x

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

def extract_img_features(img_paths, model, device):
    """
    Extracts, stores and returns image features

    Parameters:
    -----------
    img_paths: list
        the paths of images
    model: CNNModel (default: None)
      a CNNModel instance

    Return:
    --------
    numpy.ndarray
        the extracted image feature matrix from CNNModel
    """ 

    start = time()
    img_features = []

    for image_path in img_paths:
        img_features.append(
            encode_image(model, image_path, device).cpu().data.numpy()
    )

    print(f"\nGenerating set took: {hms_string(time()-start)}")

    return img_features





