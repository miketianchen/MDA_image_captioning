'''
Script part of the visualization module, not intended to be run separately

Script for 'Database Upload' tab's functionality.
    The route (url) is /database

This script will be called from `views.py`'s `database` function

This script is used to upload multiple files (images and json caption file) to AWS S3

Summary of script:
    In `view.py`, the `database` function will be initated when the user uploads
    multiple images and json file. The `database` function will temporary store
    all those files in a temporary directory at `./visualization/mda_mds/media/database_images`

    This script will then take all of those images and json file in the temporary
    directory and upload it to AWS S3.

Be sure to have your AWS S3 settings set up in STATIC_VARIABLES.json file at
    './visualization/mda_mds/mda_mds/STATIC_VARIABLES.json'
'''

import sys, json
import boto3
import time
import shutil
from botocore.exceptions import NoCredentialsError

import os

# /Users/apple/Documents/MDS_labs/DSCI_591/591_capstone_2020-mda-mds/scr/visualization/mda_mds
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Location of the temporary file directory
DATABASE_IMAGES_DIR = os.path.join(BASE_DIR, 'media/database_images')

# Location to data directory
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'data')

# Location to raw directory
RAW_PATH = os.path.join(DATA_PATH, 'raw')


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

################################################################################

folder_name = str(int(time.time()))
IMAGE_FOLDER_PATH = os.path.join(RAW_PATH, folder_name)
if not os.path.exists(IMAGE_FOLDER_PATH):
    os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)

bucket_name = STATIC_VARIABLES["S3_BUCKET_NAME"]


# UPLOAD THE IMAGES TO THE DATABASE
for filename in os.listdir(DATABASE_IMAGES_DIR):
    file_path = os.path.join(DATABASE_IMAGES_DIR, filename)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tif"):
        print(filename)
        s3_file_name = 'raw/' + folder_name + '/' + filename
        upload_to_aws(file_path, bucket_name, s3_file_name)

        shutil.move(file_path, os.path.join(IMAGE_FOLDER_PATH, filename))
    elif filename.endswith(".json"):
        json_file_name = folder_name + ".json"

        s3_file_name = 'raw/' + json_file_name
        upload_to_aws(file_path, bucket_name, s3_file_name)

        shutil.move(file_path, os.path.join(RAW_PATH, json_file_name))
    else:
        continue
