# MDA-MDS Capstone Project: Image Captioning of Earth Observation Imagery

***Creators**: Dora Qian, Fanli Zhou, James Huang, Mike Chen*

***MDS Mentor**: Varada Kolhatkar*

***MDA Partners**: Andrew Westwell-Roper, Shun Chi*

## Summary

MDA is a Canadian aerospace company, manufacturing equipment for space applications, specializing in space surveillance, space robotics, and satellite systems. MDA has access to a vast database of uncaptioned overhead satellite images, and they are interested in assigning captions to these images for image indexing, more specifically for detection of events of interest; these captions will describe objects in the photo and how they interact. In this project, we have created a complete image captioning pipeline consisting of three independent modules: a database, a deep learning model and an interactive visualization and database updating tool. 

![](imgs/dataproduct.png)

The following pipeline chart displays the workflow used in our pipeline.
![](imgs/pipeline.jpg)

## Installation instructions

**Option 1: Running the whole pipeline** 
  - An AWS S3 bucket needs to be set up as the database 
  - An AWS EC2 P3 instance needs to be set up to run the pipeline. 
  - Please follow the AWS installation instructions [here](docs/ec2_installation_steps.md)

**Option 2: Using the visualization tool with our pre-trained model and results** 
  - You can run the visualization tool locally with the following dependencies installed on your machine. 
```
python 
add dependencies here
```

## Preparing the database

We have prepared two google drive links for users to download the data. Please follow the steps below to download the data and prepare the database.

**Option 1: Running the whole pipeline** 
1. Download the data [here](), only raw data is included in the zip file. 
2. Unzip the downloaded file
3. Upload the raw data to your S3 bucket, you can either do it manually on S3 website or use the following script in your command line.
```
# make sure you replace {bucket_name} with your S3 bucket name
aws s3 sync s3://{bucket_name} data
```
4. Launch your AWS EC2 P3 instance 
5. Download this github repository by typing the following script in terminal.
```
git clone https://github.com/UBC-MDS/591_capstone_2020-mda-mds.git
``` on your terminal.
6. Sync your S3 bucket as data folder under this repository by typing

**Option 2:** To use the visualization tool only, both the raw data and final model are needed. Please download the data [here]().

2. After downloading the data, please unzip it and upload to S3 bucket.

3. Download this github repository.

4. Sync the s3 bucket with data folder under this repo usinh the command below.
```
aws s3 sync data  s3://{bucket_name}
```

## Runnning the pipeline

**Make**

You will need “GNU Make” installed on your gpu to run Make. To see if you already have it installed, type make -v into your terminal (Linux/Mac) or make --version (Windows). The version will display if you have Make installed. If you need to install it, please see the instruction [here]().  ??? is make installed on gpu? 

To clean up all the intermediate and results files, and prepare a clean environment to run the pipeline, please type the following command in terminal.
```
make clean
```

To run the whole pipeline, please type the following command in terminal.
```
make all
```

Sync the updated data folder with s3 bucket using the command below.
```
aws s3 sync s3://{bucket_name} data
```

The following usage are allowed to run any speicific part of pipeline:
```
# To prepare the data for model training
make data

# To train the model 
make train

# To generate captions for the test dataset 
make caption
make caption
```

## Running the Visualization Tool

To run the visualization tool on the EC2 instance:

1. From the root of the repo, navigate to `scr/visualization/mda_mds`, open `settings.py`

2. Add `[public domain name].ca-central-1.compute.amazonaws.com` to `ALLOWED_HOSTS`

For example:

```
ALLOWED_HOSTS = ['ec2-3-96-51-16.ca-central-1.compute.amazonaws.com', 'localhost', '127.0.0.1']
```

3. Save and exit, call

```
python manage.py runserver [public domain name].ca-central-1.compute.amazonaws.com:[port]
```

For example:

```
python manage.py runserver ec2-3-951-16.ca-central-1.compute.amazonaws.com:8443
```

You can define the port number when you launch the EC2 instance when setting the `Security Groups` by adding `Custom TCP Rule` and setting the `Port Range` to the port number.

If you launched an instance in the `sg-4a03c42a` group, then the port number is `8443`.

4. Open `http://[public domain name].ca-central-1.compute.amazonaws.com:[port]` in Chrome.

For example: `http://ec2-3-96-51-16.ca-central-1.compute.amazonaws.com:8443`
