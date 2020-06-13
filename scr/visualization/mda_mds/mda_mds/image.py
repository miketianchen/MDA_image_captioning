import sys, json
import boto3
import os
import time
import nltk

from botocore.exceptions import NoCredentialsError
from shutil import copyfile
from PIL import Image

# /Users/apple/Documents/MDS_labs/DSCI_591/591_capstone_2020-mda-mds/scr/visualization/mda_mds
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PATH for JSON CAPTION FILE
JSON_CAPTION_PATH = os.path.join(BASE_DIR, 'media/caption.json')
# PATH for MEDIA_URL
MEDIA_PATH = os.path.join(BASE_DIR, 'media/')
SCR_PATH = os.path.dirname(os.path.dirname(BASE_DIR))
EXTRACT_FEATURES_PATH = os.path.join(SCR_PATH, 'models/extract_features.py')
GENERATE_CAPTIONS_PATH = os.path.join(SCR_PATH, 'models/generate_captions.py')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'data')
RESULTS_PATH = os.path.join(DATA_PATH, 'results')

# NOTE!!! REPLACE THIS WITH ENVIRONMENT VARIABLES WHEN YOU PUSH TO GITHUB
ACCESS_KEY = 'AKIATB63UHM3M3LZZH5L'
SECRET_KEY = 'VDmCpB8e5HEjpQa8PKZlLpmulkQbjjMetTq2IFON'

# Upload data to AWS S3
def upload_to_aws(local_file, bucket, s3_file = None):
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
def model():
    # currently the picture file is saved in temp media directory, need to copy picture to data/results folder
    # copyfile(image_fullpath, os.path.join(DATA_PATH, image_name))

    extract_features_cli_call = 'python ' + str(EXTRACT_FEATURES_PATH) + ' --root_path=' + DATA_PATH + ' --output=' + image_name.split(".")[0] + ' --inputs=' + image_name
    # Example call:
    # 'python ../../../models/extract_features.py --root_path=../../../../data --output=test_rsicd_00030 --inputs=rsicd_airport_55.jpg'
    os.system(extract_features_cli_call)

    output_json_name = image_name.split(".")[0]+'_captions'
    generate_captions_cli_call = 'python ' + str(GENERATE_CAPTIONS_PATH) + ' --root_path=' + DATA_PATH + ' --inputs=' + image_name.split(".")[0] + ' --model=final_model --output=' + output_json_name
    # Example call:
    # 'python ../../../models/generate_captions.py --root_path=../../../../data --inputs=test_rsicd_00030 --model=final_model --output=test_rsicd_00030'
    os.system(generate_captions_cli_call)

    captions = read_results(output_json_name, RESULTS_PATH)

    return ['NA', captions]

def read_results(output_json_name, RESULTS_PATH):
    output_json_name = output_json_name + '.json'
    JSON_PATH = os.path.join(RESULTS_PATH, output_json_name)
    with open(JSON_PATH) as f:
        caption_dict = json.load(f)

    captions = caption_dict[image_name]
    return captions

def preprocess_image(size = (299, 299)):
    im = Image.open(image_fullpath).resize(size, Image.ANTIALIAS)
    rgb_im = im.convert('RGB')

    name = image_name[:-4]
    name = name + '.jpg'

    output_path = os.path.join(DATA_PATH, name)

    rgb_im.save(output_path, 'JPEG', quality = 95)



# upload mode; if upload mode is 'image' then only images will be uploaded
#              if upload mode is 'caption' then captions will be created and uploaded
upload_mode = sys.argv[1]

if upload_mode == "image":
    # get our data as an array from sys
    image_fullpath = sys.argv[2]
    image_name = sys.argv[3]

    if image_name.endswith('.png'):
        image_name = image_name[:-4]
        image_name = image_name + '.jpg'

    bucket_name = 'mds-capstone-mda'
    s3_images_file_name = 'upload/images/' + image_name

    preprocess_image()

    uploaded = upload_to_aws(image_fullpath, bucket_name, s3_images_file_name)

    # Return the score from the model
    output = model()
    score = output[0]
    model_caption = output[1]
    print(score+"*"+model_caption)

elif upload_mode == "caption":
    # captions
    user_caption_input = sys.argv[2]
    optional_caption_2 = sys.argv[3]
    optional_caption_3 = sys.argv[4]
    optional_caption_4 = sys.argv[5]
    optional_caption_5 = sys.argv[6]

    for filename in os.listdir(MEDIA_PATH):
        if filename.endswith(".jpg"):
            image_name = filename
            os.remove(os.path.join(MEDIA_PATH, filename))
            continue
        else:
            continue


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
        "imgid":time_stamp,
        "sentences":[]
    }}

    tokens_1 = nltk.word_tokenize(user_caption_input)
    caption[image_name]["sentences"].append({
        'raw': user_caption_input,
        'token': tokens_1,
        'sentid': int(str(time_stamp) + str(1))
    })
    tokens_2 = nltk.word_tokenize(optional_caption_2)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_2,
        'token': tokens_2,
        'sentid': int(str(time_stamp) + str(2))
    })
    tokens_3 = nltk.word_tokenize(optional_caption_3)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_3,
        'token': tokens_3,
        'sentid': int(str(time_stamp) + str(3))
    })
    tokens_4 = nltk.word_tokenize(optional_caption_4)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_4,
        'token': tokens_4,
        'sentid': int(str(time_stamp) + str(4))
    })
    tokens_5 = nltk.word_tokenize(optional_caption_5)
    caption[image_name]["sentences"].append({
        'raw': optional_caption_5,
        'token': tokens_5,
        'sentid': int(str(time_stamp) + str(5))
    })

    # captions dictionary
    # data = {}
    #
    # data[image_name] = []
    # data[image_name].append({
    #     'first_caption': user_caption_input
    # })
    # data[image_name].append({
    #     'second_caption': optional_caption_2
    # })
    # data[image_name].append({
    #     'third_caption': optional_caption_3
    # })
    # data[image_name].append({
    #     'fourth_caption': optional_caption_4
    # })
    # data[image_name].append({
    #     'fifth_caption': optional_caption_5
    # })

    with open(JSON_CAPTION_PATH, 'w') as outfile:
        json.dump(caption, outfile)

    bucket_name = 'mds-capstone-mda'
    s3_captions_file_name = 'upload/captions/' + image_name.split(".")[0] + '.json'

    caption_uploaded = upload_to_aws(JSON_CAPTION_PATH, bucket_name, s3_captions_file_name)


# # get our data as an array from sys
# image_fullpath = sys.argv[1]
# image_name = sys.argv[2]
# # captions
# user_caption_input = sys.argv[3]
# optional_caption_2 = sys.argv[4]
# optional_caption_3 = sys.argv[5]
# optional_caption_4 = sys.argv[6]
# optional_caption_5 = sys.argv[7]
# # captions dictionary
# data = {}
# data[image_name] = []
# data[image_name].append({
#     'first_caption': user_caption_input
# })
#
# if (optional_caption_2 == ""):
#     optional_caption_2 = user_caption_input
#     optional_caption_3 = user_caption_input
#     optional_caption_4 = user_caption_input
#     optional_caption_5 = user_caption_input
# elif (optional_caption_3 == ""):
#     optional_caption_3 = user_caption_input
#     optional_caption_4 = user_caption_input
#     optional_caption_5 = user_caption_input
# elif (optional_caption_4 == ""):
#     optional_caption_4 = user_caption_input
#     optional_caption_5 = user_caption_input
# else:
#     optional_caption_5 = user_caption_input
#     # throw execption here later
#
# data[image_name].append({
#     'second_caption': optional_caption_2
# })
# data[image_name].append({
#     'third_caption': optional_caption_3
# })
# data[image_name].append({
#     'fourth_caption': optional_caption_4
# })
# data[image_name].append({
#     'fifth_caption': optional_caption_5
# })
#
# with open(JSON_CAPTION_PATH, 'w') as outfile:
#     json.dump(data, outfile)



# bucket_name = 'mds-capstone-mda'
# s3_images_file_name = 'upload/images/' + image_name
# s3_captions_file_name = 'upload/captions/' + image_name.split(".")[0] + '.json'
#
# uploaded = upload_to_aws(image_fullpath, bucket_name, s3_images_file_name)
# caption_uploaded = upload_to_aws(JSON_CAPTION_PATH, bucket_name, s3_captions_file_name)
#
# # Return the score from the model
# output = model(image_fullpath, user_caption_input)
# score = output[0]
# model_caption = output[1]
# print(score+"_"+model_caption)
