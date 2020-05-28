import sys, json
import boto3
from botocore.exceptions import NoCredentialsError

# NOTE!!! REPLACE THIS WITH ENVIRONMENT VARIABLES WHEN YOU PUSH TO GITHUB
ACCESS_KEY = 'XXXXXXXXXXXX'
SECRET_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXX'

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
def model(image_path, user_caption):
    # BLAH BLAH BLAH DO THINGS...
    # Return an evaluation score
    return ['666666', 'BLAH BLAH BLAH']


#get our data as an array from sys
image_fullpath = sys.argv[1]
image_name = sys.argv[2]
user_caption_input = sys.argv[3]

#image_fullpath = './images/uploads/' + image_name
bucket_name = 'mds-capstone-mda'
s3_file_name = 'upload/' + image_name

uploaded = upload_to_aws(image_fullpath, bucket_name, s3_file_name)

# Return the score from the model
output = model(image_fullpath, user_caption_input)
score = output[0]
model_caption = output[1]
print(score+"_"+model_caption)
