all : data/interim/train.json data/processed/preprocessed_images data/processed/json/train.json data/processed/train

# split the data
data/interim/train.json data/interim/valid.json data/interim/test.json : scr/data/split_data.py
	python scr/data/split_data.py

# preprocess the images
data/processed/preprocessed_images : data/raw
	python scr/data/preprocess_image.py

# process the json files and correct the captions
data/processed/json/train.json : data/raw
	python scr/data/process_json.py
	python scr/data/correct_captions.py

# correct captions
# data/processed/json/train.json :
#	python scr/data/correct_captions.py

# sort preprocessed images into folders
data/processed/train data/processed/test data/processed/valid : data/processed/json data/processed
	python scr/data/sort_into_folders.py	

clean :
	rm -f data/interim/train.json
	rm -f data/interim/valid.json
	rm -f data/interim/test.json
	rm -r -f data/processed/preprocessed_UCM
	rm -r -f data/processed/preprocessed_RSICD
	rm -r -f data/processed/preprocessed_sydney
	rm -r -f data/processed/json
	mkdir data/processed/json
	rm -r -f data/processed/train
	rm -r -f data/processed/valid
	rm -r -f data/processed/test
