# MDA-MDS Capstone Project: Image Captioning of Earth Observation Imagery

***Creators**: Dora Qian, Fanli Zhou, James Huang, Mike Chen*

***MDS Mentor**: Varada Kolhatkar*

***MDA Partners**: Andrew Westwell-Roper, Shun Chi*

## Summary

MDA is a Canadian aerospace company, manufacturing equipment for space applications, specializing in space surveillance, space robotics, and satellite systems. MDA has access to a vast database of uncaptioned overhead satellite images, and they are interested in assigning captions to these images for image indexing, more specifically for detection of events of interest; these captions will describe objects in the photo and how they interact. In this project, we have created a complete image captioning pipeline consisting of three independent modules: a database, a deep learning model and an interactive visualization and database updating tool. 

![](imgs/dataproduct.png)

## Installation instructions

**Option 1:** To run the complete pipeline, we recommend setting up a new AWS S3 bucket as database and AWS EC2 P3 gpu instance to trian the model. Please follow the aws instructions [here](docs/ec2_installation_steps.md)

**Option 2:** To use the visualization tool only, you can use local machine with the following dependencies installed.
```
python 
add dependencies here
```

## Input Data

We have prepared two google drive links for users to download the data.

**Option 1:** To run the complete pipeline, only raw data are needed. Please download the data [here]().

**Option 2:** To use the visualization tool only, both the raw data and final model are needed. Please download the data [here]().

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

The following usage are allowed to run speicific part of pipeline:
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
