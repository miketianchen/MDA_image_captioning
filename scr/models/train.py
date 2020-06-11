# Author: Fanli Zhou
# Date: 2020-06-09

'''This script 

Usage: scr/models/train.py ROOT_PATH OUTPUT

Arguments:
ROOT_PATH         The root path of the json folder.
OUTPUT            The output trained caption model name without the filename extension.
'''

import json
from tqdm import tqdm
import pickle
from time import time
from docopt import docopt
from itertools import chain
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hms_string import hms_string
from model import CNNModel, RNNModel, CaptionModel

EPOCHS = 10

def get_embeddings(root_path, vocab_size, embedding_dim, wordtoidx):

    embeddings_index = {} 
    
    print('Loading pre-trained Glove embeddings...')

    with open(f"{root_path}/glove.6B.200d.txt", 'r', encoding='utf-8') as file:

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

def train(model, iterator, optimizer, criterion, clip, vocab_size):
    """
    train the CaptionModel

    Parameters:
    -----------
    model: CaptionModel
        a CaptionModel instance
    iterator: torch.utils.data.dataloader
        a PyTorch dataloader
    optimizer: torch.optim
        a PyTorch optimizer 
    criterion: nn.CrossEntropyLoss
        a PyTorch criterion 

    Return:
    --------
    float
        average loss
    """
    model.train()    
    epoch_loss = 0
    
    for img_features, captions in iterator:
        
        optimizer.zero_grad()

        # for each caption, the end word is not passed for training
        outputs = model(
            img_features.to(device),
            captions[:, :-1].to(device)
        )

        loss = criterion(
            outputs.view(-1, vocab_size), 
            captions[:, 1:].flatten().to(device)
        )
        epoch_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        
    return epoch_loss / len(iterator)

def init_weights(model, embedding_pretrained=True):
    """
    Initialize weights and bias in the model

    Parameters:
    -----------
    model: CaptionModel
      a CaptionModel instance
    embedding_pretrained: bool (default: True)
        not initialize the embedding matrix if True
    """  
  
    for name, param in model.named_parameters():
        if embedding_pretrained and 'embedding' in name:
            continue
        elif 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
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
            seq = [self.wordtoidx[word] for word in desc.split(' ')
                  if word in self.wordtoidx]
            # pad the sequence with 0 on the right side
            in_seq = np.pad(
                seq, 
                (0, self.max_length - len(seq)),
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

if __name__ == "__main__":

    args = docopt(__doc__)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batch_size = 200
    hidden_size = 256
    embedding_dim = 200
    
    with open( f"{args['ROOT_PATH']}/results/model_info.json", 'r') as f:
        model_info = json.load(f)
        
    embedding_matrix = get_embeddings(
        args['ROOT_PATH'],
        model_info['vocab_size'],
        embedding_dim,
        model_info['wordtoidx']
    ) 
        
    with open(f"{args['ROOT_PATH']}/results/train_descriptions.pkl", 'rb') as f:
        train_descriptions = pickle.load(f)
        
    with open(f"{args['ROOT_PATH']}/results/train.pkl", 'rb') as f:
        train_img_features = pickle.load(f)    

    train_dataset = SampleDataset(
        train_descriptions,
        train_img_features,
        model_info['wordtoidx'],
        model_info['max_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        collate_fn=my_collate
    )

    caption_model = CaptionModel(
        model_info['vocab_size'], 
        embedding_dim, 
        hidden_size=hidden_size,
        embedding_matrix=embedding_matrix, 
        embedding_train=True
    )

    init_weights(
        caption_model,
        embedding_pretrained=True
    )

    caption_model.to(device)

    # we will ignore the pad token in true target set
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        caption_model.parameters(), 
        lr=0.01
    )

    clip = 1
    start = time()
    print(f'Training...')
    for i in tqdm(range(EPOCHS * 7)):

        loss = train(
            caption_model,
            train_loader,
            optimizer,
            criterion,
            clip,
            model_info['vocab_size']
        )
        print(f'loss = {loss}')

    # reduce the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4

    for i in tqdm(range(EPOCHS * 7)):

        loss = train(
            caption_model,
            train_loader,
            optimizer,
            criterion,
            clip,
            model_info['vocab_size']
        )
        print(f'loss = {loss}')

    torch.save(caption_model, f"{args['ROOT_PATH']}/results/{args['OUTPUT']}.hdf5")
    print(f"\Training took: {hms_string(time()-start)}")
