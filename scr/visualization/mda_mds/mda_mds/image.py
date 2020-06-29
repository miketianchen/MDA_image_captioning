'''
Script part of the visualization module, not intended to be run separately

Script for 'User Image' tab's functionality
    The route (url) is /external

This script will be called from `views.py`'s `external` function

This script is used to preprocess the user uploaded image, then run the machine
learning model on the image in order to generate a caption for the image.

Summary of script:
    In `view.py`, when the `external` function is called, there will be two
    potential "upload modes" which will be initated here, depending on what the user
    does in `view.py`
        "image" mode : the image that the user uploads to the app save to S3
        "caption" mode : if the user enters the optional captions, then appropriate
            json file will be created with the user entered captions. Saves json_file
            to S3

    This script will call the machine learning model in ./data/results

Be sure to have your AWS S3 settings set up in STATIC_VARIABLES.json file at
    './visualization/mda_mds/mda_mds/STATIC_VARIABLES.json'
'''
# upload mode; if upload mode is 'image' then only images will be uploaded
#              if upload mode is 'caption' then captions will be created and uploaded

import sys, json
import boto3
import os
import time
import nltk
import shutil

from botocore.exceptions import NoCredentialsError
from PIL import Image

import torch
sys.path.append('../../models/')
from extract_features import extract_img_features
from model import CNNModel
from generate_captions import generate_caption

# /Users/apple/Documents/MDS_labs/DSCI_591/591_capstone_2020-mda-mds/scr/visualization/mda_mds
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PATH for MEDIA_URL
MEDIA_PATH = os.path.join(BASE_DIR, 'media/')
SCR_PATH = os.path.dirname(os.path.dirname(BASE_DIR))
EXTRACT_FEATURES_PATH = os.path.join(SCR_PATH, 'models/extract_features.py')
GENERATE_CAPTIONS_PATH = os.path.join(SCR_PATH, 'models/generate_captions.py')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'data')
RESULTS_PATH = os.path.join(DATA_PATH, 'results')

# PATH FOR MODEL GENENERATED CAPTIONS JSON
JSON_PATH = os.path.join(DATA_PATH, 'raw/upload_model_caption.json')

# STATIC VARIABLES
STATIC_VARIABLES_PATH = os.path.join(BASE_DIR, 'mda_mds/STATIC_VARIABLES.json')
with open(STATIC_VARIABLES_PATH) as json_file:
    STATIC_VARIABLES = json.load(json_file)

# AWS ACCESS KEY AND SECRET KEY, LOCATED IN scr/visualization/mda_mds/mda_mds
ACCESS_KEY = STATIC_VARIABLES['AWS_ACCESS_KEY']
SECRET_KEY = STATIC_VARIABLES['AWS_SECRET_ACCESS_KEY']


# Upload data to AWS S3
def upload_to_aws(local_file, bucket, s3_file = None):
    """
    Upload a singe file to the desired location in S3

    Parameters:
    _______________
    local_file: str
        path to the file in your local computer that you want to upload to S3
    bucket: str
        the name of the bucket you want to upload to. Should be set up in
        STATIC_VARIABLES.json
    s3_file: str
        the path in S3 bucket
    """
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    # If S3 object_name was not specified, use file_name
    if s3_file is None:
        s3_file = local_file

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

# Function to take the user uploaded image and run the model on it
# def model():
#     # currently the picture file is saved in temp media directory, need to copy picture to data/results folder
#     # copyfile(image_fullpath, os.path.join(DATA_PATH, image_name))
#
#     extract_features_cli_call = 'python ' + str(EXTRACT_FEATURES_PATH) + ' --root_path=' + DATA_PATH + ' --output=' + image_name.split(".")[0] + ' --inputs=' + image_name
#     # Example call:
#     # 'python ../../../models/extract_features.py --root_path=../../../../data --output=test_rsicd_00030 --inputs=rsicd_airport_55.jpg'
#     os.system(extract_features_cli_call)
#
#     output_json_name = image_name.split(".")[0]+'_captions'
#     generate_captions_cli_call = f'python {str(GENERATE_CAPTIONS_PATH)} --root_path={DATA_PATH} --inputs={os.path.splitext(image_name)[0]} --model=final_model --single=True'
#     # Example call:
#     # 'python ../../../models/generate_captions.py --root_path=../../../../data --inputs=test_rsicd_00030 --model=final_model --single=True'
#     os.system(generate_captions_cli_call)
#
#     captions = read_results(output_json_name, RESULTS_PATH)
#
#     return ['NA', captions]

def get_caption(encoder, caption_model, image_fullpath, model_info, device):
    """
    Function to take the user uploaded image and run the model on it

    Parameters:
    _______________
    encoder: CNNModel from Model
        the encoder
    caption_model: torch model
        the captioning model
    image_fullpath:
        path of the image from view.py -- sys.argv[2] "second variable passed in"
    model_info:
        from "{DATA_PATH}/results/model_info.json"
    device:
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    img_feature = extract_img_features(
        [image_fullpath],
        encoder,
        device
    )
    results = {}
    results[image_name] = generate_caption(
        caption_model,
        img_feature,
        model_info['max_length'],
        model_info['vocab_size'],
        model_info['wordtoidx'],
        model_info['idxtoword'],
        device
    )

    try:
        with open(f"{DATA_PATH}/raw/upload_model_caption.json", 'r') as fp:
            single_captions = json.load(fp)
        single_captions.update(results)
    except:
        single_captions = results

    with open(f"{DATA_PATH}/raw/upload_model_caption.json", 'w') as fp:
        json.dump(single_captions, fp)

    with open(f"{BASE_DIR}/mda_mds/image_name.json", 'w') as fp:
        json.dump({'image_name': image_name}, fp)
        
    return ['NA', results[image_name]]

# def read_results(output_json_name, RESULTS_PATH):
#     output_json_name = output_json_name + '.json'
#     with open(JSON_PATH) as f:
#         caption_dict = json.load(f)
#
#     captions = caption_dict[image_name]
#     return captions

# def preprocess_image(image_fullpath, image_name, size = (299, 299)):
#     """
#     Preprocess the image uploaded by the user

#     Parameters:
#     _______________
#     image_fullpath: str
#         path to the image
#     image_name: str
#         image name
#     size: tuple (int x, int y)
#         resize the image into x by y pixels
    
#     Return:
#     _______________
#     str
#         image name
#     """
#     im = Image.open(image_fullpath).resize(size, Image.ANTIALIAS)
#     name, extension = os.path.splitext(image_name.lower())
#     name = f'{name}.jpg'
#     output_path = os.path.join(DATA_PATH, name)
    
#     if extension != '.jpg':
#         rgb_im = im.convert('RGB')
#         rgb_im.save(output_path, 'JPEG', quality = 95)
#     else:
#         im.save(output_path, quality = 95)
#     return name    

# def relocate_image_path(image_name):
#     """
#     When the image is first uploaded, it has to be stored in a temporary folder
#     (called database_images)in the media folder (.../visualization/mda_mds/media)

#     This function will move the images to the proper folder under the data folder
#     (/591_capstone_2020-mda_mds/data/raw)

#     Parameters:
#     _______________
#     image_name: str
#         name of the image from view.py -- sys.argv[3] "third variable passed in"
#     """
#     if not os.path.exists(f'{DATA_PATH}/raw/upload'):
#         os.makedirs(f'{DATA_PATH}/raw/upload', exist_ok=True)

#     UPLOAD_PATH = os.path.join(DATA_PATH, 'raw/upload', image_name)
#     CURRENT_PATH = os.path.join(DATA_PATH, image_name)
#     shutil.move(CURRENT_PATH, UPLOAD_PATH)

def merge_two_dicts(x, y):
    """
    Merge two dictionaries

    Parameters:
    _______________
    x: dict
        first dictionary to be merged
    y: dict
        second dictionary to be merged
    """
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


# upload mode; if upload mode is 'image' then only images will be uploaded
#              if upload mode is 'caption' then captions will be created and uploaded
upload_mode = sys.argv[1]

selected_model = sys.argv[4]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load encoder
try:
    encoder = CNNModel(pretrained=True, path=f'{DATA_PATH}/vgg16.hdf5')
except:
    encoder = CNNModel(pretrained=True)
encoder.to(device)

# load the trained caption model
  
with open(f"{DATA_PATH}/results/model_info.json", 'r') as f:
         model_info = json.load(f)
try:
    caption_model = torch.load(f"{DATA_PATH}/results/{selected_model}.hdf5", map_location=device)
except:
    caption_model = torch.load(f"{DATA_PATH}/results/final_model.hdf5", map_location=device)
    

if upload_mode == "image":
    # get our data as an array from sys
    image_fullpath = sys.argv[2]
    image_name = sys.argv[3]
#     image_name = preprocess_image(image_fullpath, image_name)

    bucket_name = STATIC_VARIABLES["S3_BUCKET_NAME"]
    s3_images_file_name = 'raw/upload/' + image_name
    s3_upload_model_caption_name = 'raw/upload_model_caption.json'  

    # Return the score from the model
#     output = model()
    output = get_caption(encoder, caption_model, image_fullpath, model_info, device)
    score = output[0]
    model_caption = output[1]
    model_caption_upload = upload_to_aws(JSON_PATH, bucket_name, s3_upload_model_caption_name)

    uploaded = upload_to_aws(image_fullpath, bucket_name, s3_images_file_name)

#     relocate_image_path(image_name)

    print(score+"*"+model_caption)


elif upload_mode == "caption":
    # captions
    user_caption_input = sys.argv[2]
    optional_caption_2 = sys.argv[3]
    optional_caption_3 = sys.argv[4]
    optional_caption_4 = sys.argv[5]
    optional_caption_5 = sys.argv[6]

    with open(f"{BASE_DIR}/mda_mds/image_name.json", 'r') as fp:
        image_name = json.load(fp)['image_name']

    if (optional_caption_2 == ""):
        optional_caption_2 = user_caption_input
        optional_caption_3 = user_caption_input
        optional_caption_4 = user_caption_input
        optional_caption_5 = user_caption_input
    elif (optional_caption_3 == ""):
        optional_caption_3 = user_caption_input
        optional_caption_4 = user_caption_input
        optional_caption_5 = user_caption_input
    elif (optional_caption_4 == ""):
        optional_caption_4 = user_caption_input
        optional_caption_5 = user_caption_input
    else:
        optional_caption_5 = user_caption_input
        # throw execption here later

    time_stamp = int(time.time())

    caption = {image_name: {
        "imgid": time_stamp,
        "sentences": []
    }}

    tokens_1 = nltk.word_tokenize(user_caption_input)
    caption[image_name]["sentences"].append({
        'raw': user_caption_input,
        'tokens': tokens_1,
        'imgid':time_stamp,
        'sentid': int(str(time_stamp) + str(1))
    })
    tokens_2 = nltk.word_tokenize(optional_caption_2)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_2,
        'tokens': tokens_2,
        'imgid':time_stamp,
        'sentid': int(str(time_stamp) + str(2))
    })
    tokens_3 = nltk.word_tokenize(optional_caption_3)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_3,
        'tokens': tokens_3,
        'imgid':time_stamp,
        'sentid': int(str(time_stamp) + str(3))
    })
    tokens_4 = nltk.word_tokenize(optional_caption_4)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_4,
        'tokens': tokens_4,
        'imgid':time_stamp,
        'sentid': int(str(time_stamp) + str(4))
    })
    tokens_5 = nltk.word_tokenize(optional_caption_5)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_5,
        'tokens': tokens_5,
        'imgid':time_stamp,
        'sentid': int(str(time_stamp) + str(5))
    })

    bucket_name = STATIC_VARIABLES["S3_BUCKET_NAME"]
    s3_captions_file_name = 'raw/upload.json'

    if user_caption_input != "":
        USER_JSON_PATH = os.path.join(DATA_PATH, 'raw/upload.json')

        try:
            with open(USER_JSON_PATH, 'r') as json_file:
                user_caption_dict = json.load(json_file)

            new_caption_dict = merge_two_dicts(caption, user_caption_dict)
        except:
            new_caption_dict = caption

        with open(USER_JSON_PATH, 'w') as json_file:
            json.dump(new_caption_dict, json_file)

        caption_uploaded = upload_to_aws(USER_JSON_PATH, bucket_name, s3_captions_file_name)
