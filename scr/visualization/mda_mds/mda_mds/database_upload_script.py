import sys, json
import boto3
import time
import shutil
from botocore.exceptions import NoCredentialsError

import os

# NOTE!!! REPLACE THIS WITH ENVIRONMENT VARIABLES WHEN YOU PUSH TO GITHUB
ACCESS_KEY = 'AKIATB63UHM3M3LZZH5L'
SECRET_KEY = 'VDmCpB8e5HEjpQa8PKZlLpmulkQbjjMetTq2IFON'

# /Users/apple/Documents/MDS_labs/DSCI_591/591_capstone_2020-mda-mds/scr/visualization/mda_mds
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_IMAGES_DIR = os.path.join(BASE_DIR, 'media/database_images')

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'data')

RAW_PATH = os.path.join(DATA_PATH, 'raw')
JSON_PATH = os.path.join(DATA_PATH, 'json')

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


folder_name = str(int(time.time()))
IMAGE_FOLDER_PATH = os.path.join(RAW_PATH, folder_name)
# JSON_FOLDER_PATH = os.path.join(JSON_PATH, folder_name)
if not os.path.exists(IMAGE_FOLDER_PATH):
    os.makedirs(IMAGE_FOLDER_PATH, exist_ok=True)
# if not os.path.exists(JSON_FOLDER_PATH):
#     os.makedirs(JSON_FOLDER_PATH, exist_ok=True)

bucket_name = 'mds-capstone-mda'


# UPLOAD THE IMAGES TO THE DATABASE
for filename in os.listdir(DATABASE_IMAGES_DIR):
    # if filename.endswith(".asm") or filename.endswith(".py"):
    file_path = os.path.join(DATABASE_IMAGES_DIR, filename)
    if filename.endswith(".jpg"):
        print(filename)
        # print(file_path)
        s3_file_name = 'database_images_upload/' + filename
        upload_to_aws(file_path, bucket_name, s3_file_name)

        shutil.move(file_path, os.path.join(IMAGE_FOLDER_PATH, filename))

         # print(os.path.join(directory, filename))
    elif filename.endswith(".json"):
        s3_file_name = 'database_captions_upload/' + filename
        upload_to_aws(file_path, bucket_name, s3_file_name)


        json_file_name = folder_name + ".json"
        shutil.move(file_path, os.path.join(JSON_PATH, json_file_name))
    # try:
    #     if os.path.isfile(file_path) or os.path.islink(file_path):
    #         os.unlink(file_path)
    #     elif os.path.isdir(file_path):
    #         shutil.rmtree(file_path)
    # except Exception as e:
    #     print('Failed to delete %s. Reason: %s' % (file_path, e))
    #     continue
    else:
        continue
