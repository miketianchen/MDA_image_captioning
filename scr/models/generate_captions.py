# generate_captions.py
#
# Author: Fanli Zhou
#
# Date: 2020-06-09
#
# This script 

import json
from tqdm import tqdm
import pickle
from time import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from prepare_data import get_img_info, get_vocab, get_word_dict,\
get_embeddings, SampleDataset, my_collate, encode_image,\
extract_img_features, hms_string
from model import CNNModel, RNNModel, CaptionModel

START = "startseq"
STOP = "endseq"

def generateCaption(
    model, 
    img_features,
    max_length,
    vocab_size,
    wordtoidx,
    idxtoword,
    device
):
    in_text = START

    for i in range(max_length):

        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = np.pad(sequence, (0, max_length - len(sequence)),
                          mode='constant', constant_values=(0, 0))
        model.eval()
        yhat = model(
            torch.FloatTensor(img_features)\
            .view(-1, model.feature_size).to(device),
            torch.LongTensor(sequence).view(-1, max_length).to(device)
        )

        yhat = yhat.view(-1, vocab_size).argmax(1)
        word = idxtoword[yhat.cpu().data.numpy()[i]]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1 : -1]
    final = ' '.join(final)
    return final
