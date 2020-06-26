# Author: Fanli Zhou
# Date: 2020-06-09

'''This script extracts image features from input images.

Usage: scr/models/extract_features.py --root_path=<root_path> INPUTS ...

Arguments:
INPUTS                     The image folder names (e.g. test) or image path under the ROOT_PATH (test/rsicd_00030.jpg). The training data will be processed if this is not given.

Options:
--root_path=<root_path>    The path to the data folder which contains the raw folder.

Examples:
Case 1:
python scr/models/extract_features.py --root_path=data train

Extract feature vectors from images under the `data/train` folder,
and save outputs to `data/results/train.pkl`.

Case 2:
python scr/models/extract_features.py --root_path=data test/rsicd_00030.jpg

Extract a feature vector from the image `rsicd_00030.jpg` under the `data/test` folder,
save the feature vector to `data/results/rsicd_00030.pkl` and the image path to 
`results/rsicd_00030_paths.pkl.
'''

import os, json, pickle
from tqdm import tqdm
from time import time
import numpy as np
from PIL import Image
from docopt import docopt

import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

from model import CNNModel
from hms_string import hms_string

START = "startseq"
STOP = "endseq"

torch.manual_seed(123)
np.random.seed(123)

def encode_image(model, img_path, device):
    """
    Process the images to extract features

    Parameters:
    -----------
    model: CNNModel
      a CNNModel instance
    img_path: str
        the path of the image
    device: torch.device
        indicates whether cuda or cpu is used
        
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


def extract_img_features(img_paths, model, device):
    """
    Extracts, stores and returns image features

    Parameters:
    -----------
    img_paths: list
        the paths of images
    model: CNNModel (default: None)
        a CNNModel instance
    device: torch.device
        indicates whether cuda or cpu is used
        
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
        
    print(f"Extracting image features took: {hms_string(time()-start)}")

    return img_features

if __name__ == "__main__":

    args = docopt(__doc__)
    root_path = args['--root_path']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = CNNModel(pretrained=True, path=f'{root_path}/vgg16.hdf5')
    encoder.to(device)    
    
    for inputs in args['INPUTS']:

        output = os.path.splitext(inputs)[0].split('/')[-1]

        try:
            with open( f"{root_path}/results/model_info.json", 'r') as f:
                model_info = json.load(f)
        except:
            raise('Process the data with generate_data.py first to get "model_info.json".')

        img_paths = []

        if inputs == 'train':

            try:
                with open(f"{root_path}/results/train_paths.pkl", 'rb') as f:
                    img_paths = pickle.load(f)
            except:
                raise('Process the train data with generate_data.py first to get',
                '"train_paths.pkl".')

        else:

            try:            

                if inputs in ['train', 'valid', 'test']:
                    path = f"{root_path}/{inputs}"
                else:
                    path = f"{root_path}/preprocessed_{inputs}"
                for filename in os.listdir(path):
                    if filename.endswith('.jpg'):
                        img_paths.append(f'{path}/{filename}')
            except:
                img_paths.append(f"{root_path}/{inputs}")

            with open(f"{root_path}/results/{output}_paths.pkl", "wb") as f:
                pickle.dump(img_paths, f)

            assert os.path.isfile(f'{root_path}/results/{output}_paths.pkl'),\
                "Image paths are not saved."

        print(f'Extracting image features for {inputs} ...')
        
        img_features = extract_img_features(
            img_paths,
            encoder,
            device
        )

        with open(f"{root_path}/results/{output}.pkl", "wb") as f:
            pickle.dump(img_features, f)
        assert os.path.isfile(f'{root_path}/results/{output}.pkl'),\
            "Image features are not saved."
