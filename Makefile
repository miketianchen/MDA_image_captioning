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
all: data/json/train.json data/json/valid.json data/json/test.json \
data/json/sydney.json data/train data/test data/valid \
data/preprocessed_sydney data/results/final_model.hdf5 \
data/results/test.json data/results/sydney.json\
data/score/test_score.json data/score/test_img_score.json \
data/score/sydney_score.json data/score/sydney_img_score.json

# run all the scripts in the data preprocess stage
data: data/train data/test data/valid data/preprocessed_sydney \
data/json/train.json data/json/valid.json data/json/test.json \
data/json/sydney.json

# run all the scripts in the model training stage
train: data/results/final_model.hdf5

# run all the scripts in the captioning stage
caption: data/results/test.json data/results/sydney.json

# run all the scripts in the evaluation stage
score: data/score/test_score.json data/score/test_img_score.json \
data/score/sydney_score.json data/score/sydney_img_score.json

# process the json file for rsicd, ucm, and sydney datasets
data/json/rsicd.json data/json/ucm.json data/json/sydney.json : \
scr/data/preprocess_json.py

	python scr/data/preprocess_json.py --root_path=data rsicd ucm sydney

# combine rsicd and ucm datasets and split into train, valid and test
# datasets and correct the captions in train
data/json/train.json data/json/valid.json data/json/test.json : \
scr/data/split_data.py data/json/rsicd.json data/json/ucm.json

	python scr/data/split_data.py --root_path=data rsicd ucm

# preprocess and sort the images for training and save under 
# train/valid/test folders
data/train data/test data/valid : \
scr/data/preprocess_image.py data/json/train.json \
data/json/valid.json data/json/test.json

	python scr/data/preprocess_image.py --root_path=data \
    raw/ucm raw/rsicd --train=True

    
# preprocess sydney images and save under the preprocessed_sydney folder
data/preprocessed_sydney : \
scr/data/preprocess_image.py

	python scr/data/preprocess_image.py --root_path=data raw/sydney
    
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
data/results/train_paths.pkl data/train data/valid

	python scr/models/extract_features.py --root_path=data --output=train

# train the caption model
data/results/final_model.hdf5 : \
scr/models/train.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/results/train_descriptions.pkl data/results/train.pkl 

	python scr/models/train.py --root_path=data --output=final_model

# extract image features from test images
data/results/test.pkl data/results/test_paths.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/test

	python scr/models/extract_features.py \
    --root_path=data --output=test --inputs=test

# generate captions for the test images
data/results/test.json : \
scr/models/generate_captions.py scr/models/hms_string.py \
data/results/test.pkl data/results/test_paths.pkl \
data/results/final_model.hdf5

	python scr/models/generate_captions.py \
    --root_path=data --inputs=test --model=final_model --output=test

# extract imgae features from the sydney images
data/results/sydney.pkl data/results/sydney_paths.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/preprocessed_sydney

	python scr/models/extract_features.py \
    --root_path=data --output=sydney --inputs=preprocessed_sydney

# generate captions for sydney images
data/results/sydney.json : \
scr/models/generate_captions.py scr/models/hms_string.py \
data/results/sydney.pkl data/results/sydney_paths.pkl \
data/results/final_model.hdf5

	python scr/models/generate_captions.py \
    --root_path=data --inputs=sydney --model=final_model --output=sydney
    
# evaluate the model generated captions for test images
data/score/test_score.json data/score/test_img_score.json : \
data/results/test.json data/json/test.json scr/evaluation/eval_model.py

	python scr/evaluation/eval_model.py --root_path=data --inputs=test

# evaluate the model generated captions for sydney images
data/score/sydney_score.json data/score/sydney_img_score.json : \
data/results/sydney.json data/json/sydney.json scr/evaluation/eval_model.py

	python scr/evaluation/eval_model.py --root_path=data --inputs=sydney

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
	rm -rf data/score