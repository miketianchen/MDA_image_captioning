
# Author: Fanli Zhou
# Date: 2020-06-09

'''This script generates captions for input images.

Usage: scr/models/generate_captions.py --root_path=<root_path> INPUTS ... --model=<model> [--single=<single>]

Arguments:
INPUTS                     The input image feature file names (no extension).

Options:
--root_path=<root_path>    The path to the data folder which contains the raw folder.
--model=<model>            The trained caption model names (no extension).
--single=<single>          Save the caption to `raw/upload_model_caption.json` or not [default: False].


Examples:
Case 1:
python scr/models/generate_captions.py --root_path=data test --model=final_model

Take feature vectors from `data/results/test.pkl` to generate captions with the model saved in 
`data/results/final_model.hdf5`, and save outputs to `data/json/test_model_caption.json`.

Case 2:
python scr/models/generate_captions.py --root_path=data single_feature_vector --model=final_model --single=True

Take the feature vector from `data/results/single_feature_vector.pkl` to generate captions with the model saved in 
`data/results/final_model.hdf5`, and save outputs in `data/raw/upload_model_caption.json`.
'''

import os, json, pickle
from tqdm import tqdm
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
    idxtoword, 
    device
):
    """
    Generates a caption

    Parameters:
    -----------
    model: CaptionModel
        a CaptionModel instance
    img_features: numpy.ndarray
        the image features
    max_length: int
        the maximum length of the generated caption
    wordtoidx: dict
        the dict to get word index
    idxtoword: dict
        the dict to get word
    device: torch.device
        indicates whether cuda or cpu is used

    Return:
    --------
    str
        the generated caption
    """ 
        
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
    root_path = args['--root_path']
    model = args['--model']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        with open( f"{root_path}/results/model_info.json", 'r') as f:
            model_info = json.load(f)

        caption_model = torch.load(f"{root_path}/results/{model}.hdf5", map_location=device)  
    except:
        raise('Please train the model first.')
    
    for inputs in args['INPUTS']:

        try:
            with open(f"{root_path}/results/{inputs}.pkl", 'rb') as f:
                img_features = pickle.load(f)
            with open(f"{root_path}/results/{inputs}_paths.pkl", 'rb') as f:
                img_paths = pickle.load(f)
        except:
            raise('Process the data with extract_features.py first.')

        # generate results
        results = {}
        print(f'Generating captions for {inputs}...')
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
                model_info['idxtoword'], 
                device
            )

        if args['--single'] == 'False':

            with open(f"{root_path}/json/{inputs}_model_caption.json", 'w') as fp:
                json.dump(results, fp)

            assert os.path.isfile(f'{root_path}/json/{inputs}_model_caption.json'),\
            "Captions are not saved."

        else:
            try:
                with open(f"{root_path}/raw/upload_model_caption.json", 'r') as fp:
                    single_captions = json.load(fp)
                single_captions.update(results)
            except:
                single_captions = results

            with open(f"{root_path}/raw/upload_model_caption.json", 'w') as fp:
                json.dump(single_captions, fp)

            assert os.path.isfile(f'{root_path}/raw/upload_model_caption.json'),\
            "Captions are not saved."

        print(f"Generating captions took: {hms_string(time()-start)}")
