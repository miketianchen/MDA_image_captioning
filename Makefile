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
data/json/test_model_caption.json data/json/sydney_model_caption.json\
data/score/test_score.json data/score/test_img_score.json \
data/score/sydney_score.json data/score/sydney_img_score.json

# run all the scripts in the data preprocess stage
data: data/train data/test data/valid data/preprocessed_sydney \
data/json/train.json data/json/valid.json data/json/test.json \
data/json/sydney.json

# run all the scripts in the model training stage
train: data/results/final_model.hdf5

# run all the scripts in the captioning stage
caption: data/json/test_model_caption.json data/json/sydney_model_caption.json

# run all the scripts in the evaluation stage
score: data/score/test_score.json data/score/test_img_score.json \
data/score/sydney_score.json data/score/sydney_img_score.json

# set root path to the data folder which contains the raw folder
root_path := data

# process the json file for rsicd, ucm, and sydney datasets
data/json/rsicd.json data/json/ucm.json data/json/sydney.json : \
scr/data/preprocess_json.py

	python scr/data/preprocess_json.py --root_path=$(root_path) rsicd ucm sydney

# combine rsicd and ucm datasets and split into train, valid and test
# datasets and correct the captions in train
data/json/train.json data/json/valid.json data/json/test.json : \
scr/data/split_data.py data/json/rsicd.json data/json/ucm.json

	python scr/data/split_data.py --root_path=$(root_path) rsicd ucm

# preprocess and sort the images for training and save under 
# train/valid/test folders
data/train data/test data/valid : \
scr/data/preprocess_image.py data/json/train.json \
data/json/valid.json data/json/test.json

	python scr/data/preprocess_image.py --root_path=$(root_path) \
    raw/ucm raw/rsicd --train=True

    
# preprocess sydney images and save under the preprocessed_sydney folder
data/preprocessed_sydney : \
scr/data/preprocess_image.py

	python scr/data/preprocess_image.py --root_path=$(root_path) raw/sydney
    
# combine train and valid data as the training data 
# and prepare data for training
data/results/train_paths.pkl data/results/train_descriptions.pkl \
data/results/model_info.json : \
scr/models/prepare_data.py data/json/train.json data/json/valid.json

	python scr/models/prepare_data.py --root_path=$(root_path) train valid

# extract image features from training data
data/results/train.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/results/train_paths.pkl data/train data/valid

	python scr/models/extract_features.py --root_path=$(root_path) \
    --output=train

# train the caption model
data/results/final_model.hdf5 : \
scr/models/train.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/results/train_descriptions.pkl data/results/train.pkl 

	python scr/models/train.py --root_path=$(root_path) --output=final_model

# extract image features from test images
data/results/test.pkl data/results/test_paths.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/test

	python scr/models/extract_features.py \
    --root_path=$(root_path) --output=test --inputs=test

# generate captions for the test images
data/json/test_model_caption.json : \
scr/models/generate_captions.py scr/models/hms_string.py \
data/results/test.pkl data/results/test_paths.pkl \
data/results/final_model.hdf5

	python scr/models/generate_captions.py --root_path=$(root_path) \
    --inputs=test --model=final_model --output=test

# extract imgae features from the sydney images
data/results/sydney.pkl data/results/sydney_paths.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/preprocessed_sydney

	python scr/models/extract_features.py --root_path=$(root_path) \
    --output=sydney --inputs=preprocessed_sydney

# generate captions for sydney images
data/json/sydney_model_caption.json : \
scr/models/generate_captions.py scr/models/hms_string.py \
data/results/sydney.pkl data/results/sydney_paths.pkl \
data/results/final_model.hdf5

	python scr/models/generate_captions.py --root_path=$(root_path) \
    --inputs=sydney --model=final_model --output=sydney
    
# evaluate the model generated captions for test images
data/score/test_score.json data/score/test_img_score.json : \
data/json/test_model_caption.json data/json/test.json scr/evaluation/eval_model.py

	python scr/evaluation/eval_model.py --root_path=$(root_path) \
    --inputs=test

# evaluate the model generated captions for sydney images
data/score/sydney_score.json data/score/sydney_img_score.json : \
data/json/sydney_model_caption.json data/json/sydney.json scr/evaluation/eval_model.py

	python scr/evaluation/eval_model.py --root_path=$(root_path) \
    --inputs=sydney

# Clean up intermediate and results files
clean : 
	rm -rf $(root_path)/json/train.json
	rm -rf $(root_path)/json/test.json
	rm -rf $(root_path)/json/valid.json
	rm -rf $(root_path)/json/ucm.json
	rm -rf $(root_path)/json/rsicd.json
	rm -rf $(root_path)/json/sydney.json
	rm -rf $(root_path)/json/test_model_caption.json
	rm -rf $(root_path)/json/sydney_model_caption.json
	rm -rf $(root_path)/preprocessed_sydney
	rm -rf $(root_path)/preprocessed_ucm
	rm -rf $(root_path)/preprocessed_rsicd
	rm -rf $(root_path)/test
	rm -rf $(root_path)/train
	rm -rf $(root_path)/valid
	rm -rf $(root_path)/results
	rm -rf $(root_path)/score