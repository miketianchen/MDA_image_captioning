# Author: Fanli Zhou
# Date: 2020-06-09

'''This script 

Usage: scr/models/extract_features.py ROOT_PATH OUTPUT [--inputs=<inputs>]

Arguments:
ROOT_PATH         The root path of the json folder (e.g. ../s3).
OUTPUT            The output file name (e.g. test).

Options:
--inputs=<inputs> The image folder name (e.g. test) or image path under the ROOT_PATH (test/rsicd_00030.jpg). The training data will be processed if this is not given.
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


def encode_image(model, img_path):
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


def extract_img_features(img_paths, model):
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
    print(f'Extracting image features ...')
    start = time()
    img_features = []

    for image_path in img_paths:
        img_features.append(
            encode_image(model, image_path).cpu().data.numpy()
    )
        
    print(f"Extracting image features took: {hms_string(time()-start)}")

    return img_features

if __name__ == "__main__":

    args = docopt(__doc__)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
    with open( f"{args['ROOT_PATH']}/results/model_info.json", 'r') as f:
        model_info = json.load(f)
    
    img_paths = []

    if args['--inputs'] is None:
        with open(f"{args['ROOT_PATH']}/results/train_paths.pkl", 'rb') as f:
            img_paths = pickle.load(f)

    else:
        path = f"{args['ROOT_PATH']}/{args['--inputs']}"
        try:            
            for filename in os.listdir(path):
                if filename.endswith('.jpg'):
                    img_paths.append(path + f'/{filename}')
        except:
            img_paths.append(path)
            
        with open(f"{args['ROOT_PATH']}/results/{args['OUTPUT']}_paths.pkl", "wb") as f:
            pickle.dump(img_paths, f)
        
    encoder = CNNModel(pretrained=True)
    encoder.to(device)
    
    img_features = extract_img_features(
        img_paths,
        encoder
    )
    with open(f"{args['ROOT_PATH']}/results/{args['OUTPUT']}.pkl", "wb") as f:
        pickle.dump(img_features, f)

