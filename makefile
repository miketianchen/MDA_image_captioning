# Make file for UBC MDS-MDA Capstone Project
# Dora Qian, Fanli Zhou, James Huang, Mike Chen
# June 9, 2020
# 
# This driver script completes xxx, This script takes no arguments. 
# 
# usage: make all  
#					 to run all the pipeline
# usage: make data  
#					 to prepare the data for model training 
# usage: make train  
#					 to train the model 
# usage: make caption  
#					 to generate captions for the test dataset 
# usage: make clean 
#					 to clean up all the intermediate and results files

# run all the steps in pipeline
all: data/results/test.json

# run all the scripts in the data preprocess stage
data: data/train data/test data/valid data/json data/preprocessed_sydney

# run all the scripts in the model training stage
train: data/results/final_model.hdf5

caption: data/results/test.json

# split the dataset into train/valid/test, process the json 
# file and correct the captions in train json
data/json/train.json data/json/valid.json data/json/test.json : \
scr/data/preprocess_json.py

	python scr/data/preprocess_json.py --input_path="data/raw" \
	--output_path="data/json"

# preprocess the images and save under train/valid/test folders
data/preprocessed_ucm data/preprocessed_rsicd data/preprocessed_sydney : \
scr/data/preprocess_image.py

	python scr/data/preprocess_image.py --input_path="data/raw" 

# sort the preprocessed images into corresponding train/valid/test folders
data/train data/test data/valid : \
data/preprocessed_ucm data/preprocessed_rsicd \
data/preprocessed_sydney data/json/train.json \
data/json/valid.json data/json/test.json scr/data/sort_images.py

	python scr/data/sort_images.py --json_path="data/json" \
	--img_path="data" --output_path="data"

# combine train and valid data as the training data 
# and prepare data for training
data/results/train_paths.pkl data/results/train_descriptions.pkl \
data/results/model_info.json : \
scr/models/prepare_data.py data/json/train.json data/json/valid.json

	python scr/models/prepare_data.py --root_path=data train valid

# extract image features from training data
data/results/train.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/results/train_paths.pkl

	python scr/models/extract_features.py --root_path=data --output=train

# train the caption model
data/results/final_model.hdf5 : \
scr/models/train.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/results/train_descriptions.pkl data/results/train.pkl 

	python scr/models/train.py --root_path=data --output=final_model

# extract imgae features from the test images
data/results/test.pkl data/results/test_paths.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json

	python scr/models/extract_features.py \
    --root_path=data --output=test --inputs=test

# generate captions for the test images
data/results/test.json : \
scr/models/generate_captions.py scr/models/hms_string.py \
data/results/test.pkl data/results/test_paths.pkl \
data/results/final_model.hdf5

	python scr/models/generate_captions.py \
    --root_path=data --inputs=test --model=final_model --output=test


# Clean up intermediate and results files
clean : 
	rm -rf data/json
	rm -rf data/preprocessed_sydney
	rm -rf data/preprocessed_ucm
	rm -rf data/preprocessed_rsicd
	rm -rf data/test
	rm -rf data/train
	rm -rf data/valid
	rm -rf data/results