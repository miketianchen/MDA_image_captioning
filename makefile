# Make file for UBC MDS-MDA Capstone Project
# Dora Qian, Fanli Zhou, James Huang, Mike Chen
# June 9, 2020
# 
# This driver script completes xxx, This script takes no arguments. 
# 
# usage: make all  
#							to run all the pipeline
# usage: make data  
#							to prepare the data for model training 
# usage: make clean 
#							to clean up all the intermediate and results files

# run all the steps in pipeline

# run all the scripts in the data preprocess stage
data: data/train data/test data/valid data/json data/preprocessed_sydney

# split the dataset into train/valid/test, process the json file and correct the captions in train json
data/json/train.json data/json/valid.json data/json/test.json : scr/data/preprocess_json.py
	python scr/data/preprocess_json.py --input_path="data/raw" --output_path="data/json"

# preprocess the images and save under train/valid/test folders
data/preprocessed_ucm data/preprocessed_rsicd data/preprocessed_sydney : scr/data/preprocess_image.py
	python scr/data/preprocess_image.py --input_path="data/raw" 

# sort the preprocessed images into corresponding train/valid/test folders
data/train data/test data/valid : data/preprocessed_ucm data/preprocessed_rsicd data/preprocessed_sydney data/json/train.json data/json/valid.json data/json/test.json scr/data/sort_images.py
	python scr/data/sort_images.py --json_path="data/json" --img_path="data" --output_path="data"
	
# Clean up intermediate and results files
clean : 
	rm -rf data/json
	rm -rf data/preprocessed_sydney
	rm -rf data/preprocessed_ucm
	rm -rf data/preprocessed_rsicd
	rm -rf data/test
	rm -rf data/train
	rm -rf data/valid