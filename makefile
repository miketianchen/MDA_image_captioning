all : data/processed/train

# split the data
data/interim : data/raw
	python scr/data/split_data.py

# preprocess the images
data/processed/marker : data/raw
	python scr/data/preprocess_image.py

# process the json files and correct the captions
data/processed/json/train.json : data/interim
	python scr/data/process_json.py
	python scr/data/correct_captions.py

# sort preprocessed images into folders
data/processed/train data/processed/valid data/processed/test : data/processed/json/train.json data/processed/marker
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
	rm -r -f data/processed/marker
