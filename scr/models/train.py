# train.py
#
# Author: Fanli Zhou
#
# Date: 2020-06-09
#
# This script defines train and init_weights


import torch
import torch.nn as nn


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
            

