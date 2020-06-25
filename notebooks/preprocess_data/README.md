### Data Preprocessing

This folder contains notebooks used to create scripts which preprocess the data (both the image files (jpg) and captions (json))

|Notebook| Description|
|---|---|
|`2-dq-eda.ipynb`|Notebook for initial EDA|
|`3-mc-preprocess_image.ipynb`|Notebook for converting all of the image file types to jpg, and to the same size as well as retaining 95% image quality|
|`4-mc-combine_sets_split_sets.ipynb`|Notebook for spliting the JSON caption files into test/train/valid sets|
|`4.1-dq-correct_problematic_captions.ipynb`|Changed faulty captions, data cleaning|
|`5-mc-place_image_in_correct.ipynb`|Notebook for placing all of the images into the correct files according to how `4-mc-combine_sets_split_sets.ipynb` split the caption json file|
|`23-mc-preprocess_sydney_captions.ipynb`|Notebook for preprocessing our cross-validation dataset the Syndey dataset|
