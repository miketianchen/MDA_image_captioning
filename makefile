# Make file for UBC MDS-MDA Capstone Project
# Dora Qian, Fanli Zhou, James Huang, Mike Chen
# June 9, 2020
# 
# This driver script completes xxx, This script takes no arguments. 
# 
# usage: make all  
#							to run all the pipeline
# usage: make clean 
#							to clean up all the intermediate and results files

# run all the scripts in the data preprocess stage
data: data/json/train.json data/json/valid.json data/json/test.json

# split the dataset into train/valid/test, process the json file and correct the captions in train json
data/json/train.json data/json/valid.json data/json/test.json : scr/data/process_json.py
	python scr/data/process_json.py --input_path="data/raw" --output_path="data/json"

# Clean up intermediate and results files
clean : 
	rm -f data/json/*.json