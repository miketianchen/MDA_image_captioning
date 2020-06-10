# Author: Fanli Zhou
# Date: 2020-06-09

'''This script 

Usage: src/model/train.py --input=<input> --output=<output>

Options:
--input=<input>  The root path of inputs.
--output=<output>  The output model name. 
'''

import json
from tqdm import tqdm
import pickle
from time import time
from docopt import docopt
from itertools import chain

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_data import get_img_info, get_vocab, get_word_dict,\
get_embeddings, SampleDataset, my_collate, encode_image,\
extract_img_features, hms_string
from model import CNNModel, RNNModel, CaptionModel

EPOCHS = 10


opt = docopt(__doc__)

def train(model, iterator, optimizer, criterion, clip, vocab_size, device):
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

    root_path = opt["--input"]

    train_paths, train_descriptions, max_length_train =\
    get_img_info(root_path, 'train')
    valid_paths, valid_descriptions, max_length_valid =\
    get_img_info(root_path, 'valid')
    
    print(f'{len(train_paths)} images from for training')
    
    train_paths.extend(valid_paths.copy())
    train_descriptions.extend(valid_descriptions.copy())
    # add a start and stop token at the beginning/end
    for v in train_descriptions:
        for d in range(len(v)):
            v[d] = f'{START} {v[d]} {STOP}'

    max_length = max(max_length_train, max_length_valid) + 2

    vocab = get_vocab(train_descriptions, word_count_threshold=10)
    idxtoword, wordtoidx = get_word_dict(vocab)

    vocab_size = len(idxtoword) + 1
    batch_size = 200
    hidden_size = 256
    embedding_dim = 200
    cnn_type = 'vgg16'


    embedding_matrix = get_embeddings(
        root_path,
        vocab_size,
        embedding_dim,
        wordtoidx
    ) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_info = {
        'max_length': max_length,
        'idxtoword': idxtoword,
        'wordtoidx': wordtoidx,
        'vocab_size': vocab_size,
    }

    with open( f'{root_path}/model/model_info.json', 'w') as file:
        json.dump(model_info)

    encoder = CNNModel(cnn_type, pretrained=True)
    encoder.to(device)
    
    train_img_features = extract_img_features(
        'training',
        train_paths,
        encoder, 
        device
    )

    train_dataset = SampleDataset(
        train_descriptions,
        train_img_features,
        wordtoidx,
        max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        collate_fn=my_collate
    )


    caption_model = CaptionModel(
        cnn_type, 
        vocab_size, 
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
            vocab_size,
            device
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
            vocab_size,
            device
        )
        print(f'loss = {loss}')

    torch.save(caption_model, f'{root_path}/model/{opt["--output"]}')
    print(f"\Training took: {hms_string(time()-start)}")
