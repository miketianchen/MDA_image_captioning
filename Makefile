# Make file for UBC MDS-MDA Capstone Project
# Dora Qian, Fanli Zhou, James Huang, Mike Chen
# June 9, 2020
# 
# This driver script completes the whole pipeline and takes no arguments. 
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


####################################################################################
####################################################################################


# set root path to the data folder which contains the raw folder
root_path := data
# set the name of the trained model
final_model := final_model
# define the json files to process 
json_to_process := rsicd ucm sydney
# define the json file to combine for train/valid/test split
combine_set := rsicd ucm
# define the image folders to combine for train/valid/test split
combine_img := raw/ucm raw/rsicd
# define the image folders to preprocess
img_to_process := raw/sydney
# define the datasets to combine for training
train_set := train valid
# define the datasets to test separately
test_set := test sydney

####################################################################################
####################################################################################


# run all the steps in pipeline
all: data/json/train.json data/json/valid.json data/json/test.json \
data/json/sydney.json data/train data/test data/valid \
data/preprocessed_sydney data/results/$(final_model).hdf5 \
data/json/test_model_caption.json data/json/sydney_model_caption.json\
data/score/test_score.json data/score/test_img_score.json \
data/score/sydney_score.json data/score/sydney_img_score.json

# run all the scripts in the data preprocess stage
data: data/train data/test data/valid data/preprocessed_sydney \
data/json/train.json data/json/valid.json data/json/test.json \
data/json/sydney.json

# run all the scripts in the model training stage
train: data/results/$(final_model).hdf5

# run all the scripts in the captioning stage
caption: data/json/test_model_caption.json data/json/sydney_model_caption.json

# run all the scripts in the evaluation stage
score: data/score/test_score.json data/score/test_img_score.json \
data/score/sydney_score.json data/score/sydney_img_score.json


####################################################################################
####################################################################################


# process the json file for rsicd, ucm, and sydney datasets
data/json/rsicd.json data/json/ucm.json data/json/sydney.json : \
scr/data/preprocess_json.py

	python scr/data/preprocess_json.py --root_path=$(root_path) $(json_to_process) 

# combine rsicd and ucm datasets and split into train, valid and test
# datasets and correct the captions in train
data/json/train.json data/json/valid.json data/json/test.json : \
scr/data/split_data.py data/json/rsicd.json data/json/ucm.json

	python scr/data/split_data.py --root_path=$(root_path) $(combine_set)

# preprocess and sort the images for training and save under 
# train/valid/test folders
data/train data/test data/valid : \
scr/data/preprocess_image.py data/json/train.json \
data/json/valid.json data/json/test.json

	python scr/data/preprocess_image.py --root_path=$(root_path) \
    $(combine_img) --train=True

    
# preprocess sydney images and save under the preprocessed_sydney folder
data/preprocessed_sydney : \
scr/data/preprocess_image.py

	python scr/data/preprocess_image.py --root_path=$(root_path) $(img_to_process)

####################################################################################

# combine train and valid data as the training data 
# and prepare data for training
data/results/train_paths.pkl data/results/train_descriptions.pkl \
data/results/model_info.json : \
scr/models/prepare_data.py data/json/train.json data/json/valid.json

	python scr/models/prepare_data.py --root_path=$(root_path) $(train_set)

# extract image features from training data
data/results/train.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/results/train_paths.pkl data/train data/valid

	python scr/models/extract_features.py --root_path=$(root_path) train

# train the caption model
data/results/$(final_model).hdf5 : \
scr/models/train.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/results/train_descriptions.pkl data/results/train.pkl 

	python scr/models/train.py --root_path=$(root_path) --output=$(final_model)

####################################################################################

# extract image features from test and sydney images
data/results/test.pkl data/results/test_paths.pkl \
data/results/sydney.pkl data/results/sydney_paths.pkl : \
scr/models/extract_features.py scr/models/model.py \
scr/models/hms_string.py data/results/model_info.json \
data/test data/preprocessed_sydney

	python scr/models/extract_features.py \
    --root_path=$(root_path) $(test_set)

# generate captions for the test and sydney images
data/json/test_model_caption.json data/json/sydney_model_caption.json : \
scr/models/generate_captions.py scr/models/hms_string.py \
data/results/test.pkl data/results/test_paths.pkl \
data/results/sydney.pkl data/results/sydney_paths.pkl \
data/results/$(final_model).hdf5

	python scr/models/generate_captions.py --root_path=$(root_path) \
    $(test_set) --model=$(final_model)
    
####################################################################################

# evaluate the model generated captions for test images
data/score/test_score.json data/score/test_img_score.json \
data/score/sydney_score.json data/score/sydney_img_score.json : \
data/json/test_model_caption.json data/json/test.json \
data/json/sydney_model_caption.json data/json/sydney.json \
scr/evaluation/eval_model.py

	python scr/evaluation/eval_model.py --root_path=$(root_path) $(test_set)


####################################################################################
####################################################################################


# Clean up intermediate and results files
clean : 
	rm -rf $(root_path)/json
	rm -rf $(root_path)/preprocessed_*
	rm -rf $(root_path)/test
	rm -rf $(root_path)/train
	rm -rf $(root_path)/valid
	rm -rf $(root_path)/results
	rm -rf $(root_path)/score
