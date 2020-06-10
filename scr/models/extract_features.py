# Author: Fanli Zhou
# Date: 2020-06-09

'''This script 

Usage: src/model/prepare_data.py [--train=<train>] [--single=<single>] INPUT_PATH NAMES ...

Arguments:
INPUTS               The path for the file or folder

Options:
--train=<train>    Prepare data for training or not [default: False].
--single=<single>  Prepare a single images or not [default: False]. 
'''

import json
from tqdm import tqdm
import pickle
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

START = "startseq"
STOP = "endseq"
opt = docopt(__doc__)



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

def extract_img_features(name, img_paths, model, device):
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

    print(f"Extracting image features for the {name} set took: {hms_string(time()-start)}")

    return img_features

if __name__ == "__main__":

    root_path = opt["--input"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = CNNModel(cnn_type, pretrained=True)
    encoder.to(device)
    
    train_img_features = extract_img_features(
        'training',
        train_paths,
        encoder, 
        device
    )
