# Author: Fanli Zhou
# Date: 2020-06-09

'''This script

Usage: scr/models/generate_captions.py --root_path=<root_path> --inputs=<inputs> --model=<model> --output=<output>

Options:
--root_path=<root_path>    The path to the data folder which contains the raw folder.
--inputs=<inputs>          The input image feature file name (no extension).
--model=<model>            The trained caption model name (no extension).
--output=<output>          The output file name (no extension).
'''

import json
from tqdm import tqdm
import pickle
from time import time
from docopt import docopt
import numpy as np

import torch

from hms_string import hms_string

START = "startseq"
STOP = "endseq"

torch.manual_seed(123)
np.random.seed(123)

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

    opt = docopt(__doc__)
    root_path = opt['--root_path']
    inputs = opt['--inputs']
    model = opt['--model']
    output = opt['--output']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open( f"{root_path}/results/model_info.json", 'r') as f:
        model_info = json.load(f)

    with open(f"{root_path}/results/{inputs}.pkl", 'rb') as f:
        img_features = pickle.load(f)
    with open(f"{root_path}/results/{inputs}_paths.pkl", 'rb') as f:
        img_paths = pickle.load(f)

    caption_model = torch.load(f"{root_path}/results/{model}.hdf5", map_location=device)



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

    with open(f"{root_path}/results/{output}.json", 'w') as fp:
        json.dump(results, fp)

    print(f"Generating captions took: {hms_string(time()-start)}")
