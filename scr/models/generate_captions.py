# Author: Fanli Zhou
# Date: 2020-06-09

'''This script 

Usage: scr/models/generate_captions.py ROOT_PATH INPUT MODEL OUTPUT

Arguments:
ROOT_PATH         The root path of the json folder.
INPUT             The input image feature file name without the filename extension.
MODEL             The trained caption model name without the filename extension.
OUTPUT            The output file name without the filename extension.
'''

import json
from tqdm import tqdm
import pickle
from time import time
from docopt import docopt
import numpy as np

import torch

from extract_features import hms_string

START = "startseq"
STOP = "endseq"

def generate_caption(
    model, 
    img_features,
    max_length,
    vocab_size,
    wordtoidx,
    idxtoword
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
        word = idxtoword[str(yhat.cpu().data.numpy()[i])]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1 : -1]
    final = ' '.join(final)
    return final

if __name__ == "__main__":

    args = docopt(__doc__)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open( f"{args['ROOT_PATH']}/results/model_info.json", 'r') as f:
        model_info = json.load(f)
        
    with open(f"{args['ROOT_PATH']}/results/{args['INPUT']}.pkl", 'rb') as f:
        img_features = pickle.load(f)  
    with open(f"{args['ROOT_PATH']}/results/{args['INPUT']}_paths.pkl", 'rb') as f:
        img_paths = pickle.load(f)  
        
    caption_model = torch.load(f"{args['ROOT_PATH']}/results/{args['MODEL']}.hdf5")    
    
    # generate results
    results = {}
    print(f'Generating captions...')
    start = time()

    for n in tqdm(range(len(img_paths))):
        # note the filename splitting depends on path
        filename = img_paths[n].split('/')[-1]
        results[filename] = generate_caption(
            caption_model, 
            img_features[n],
            model_info['max_length'],
            model_info['vocab_size'],
            model_info['wordtoidx'],
            model_info['idxtoword']
        )

    with open(f"{args['ROOT_PATH']}/results/{args['OUTPUT']}.json", 'w') as fp:
        json.dump(results, fp)    

    print(f"\Training took: {hms_string(time()-start)}")
