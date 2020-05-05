## Executive Summary

In our MDS-MDA joint Capstone project, we aim to caption overhead Earth observation images captured by satellites.
 
## Introduction

MDA is an aerospace company, and has access to a vast database of the aforementioned satellite images. These photos in a vacuum and without context do not offer very much information on their own, as people naturally query things with words. Extracting a caption from an image makes it much more accessible. These captions can be used to tag and sort images based on their content, return a search query, and evaluate similarity between images, for example. Adding captions to images is not a binary problem; with a wide range of quality for captions, we expect this to be a complex problem with a complex solution.
Since MDA's images are all not yet captioned, we will be using a few public datasets containing captioned images of overhead satellite images to train our model first, before testing them on a manually labelled database of MDA images. We will likely start with a model using the encoder-decoder architecture, though we may attempt to tune a pre-trained model given the circumstances.

## Data Product Description

The final data product is a complete image captioning pipeline consisting of three independent components: a database, a deep learning model and an interactive visualization dashboard. First, the non-SQL database is used to store all the remote sensing images and associated captions. We would start by creating two separate folders storing images and JSON files for easy-extracting purposes. Second, the deep learning model would be python-based and implemented using CNN and LSTM. With this model, we would be able to train the remote sensing images and predict the accurate captions. The model would also be easy to maintain and update. Last, a Dash-based visualization would allow users to interact with the model. The users would be able to get one or multiple random sampled images from the database and their predicted captions. Moreover, users could also choose to upload images outside the database to get the predicted captions.
 
[insert pipeline image here]
 
## Data Description 

In order to train our model, we have three labeled datasets. The three labeled datasets are UCM_captions, RSICD and Sydney_captions. 

The UCM_captions dataset is based off of the “UC Merced Land Use Dataset”. It contains land-uses satellite images. There are 21 different types of classes of images ranging from airplane fields, baseball diamond to overpass and runways. There are 100 images in every class, and each image has a resolution of 256 X 256 pixels. 

The Sydney_captions dataset is  extracted from a large 18000 X 14000 pixel image of Sydney via Google Earth. The images in the dataset are selected and cropped from the original Google Earth image.There are 7 different classes of images in this dataset, which comprises of residential, airport, river, oceans, meadow, industrial and runway images. 

The RSICD dataset (Remote Sensing Imaging Captioning Dataset) is the state of the art dataset, which contains images captured from airplanes/satellites. The captions are sourced from volunteers, and every image will include 5 different captions, from 5 different volunteers to ensure diversity of the caption.  

Every single one of the dataset above, includes a `json` file which contains all of the image captions. In addition, a `rar` compressed file which contains all of the images. 

## Data Science Techniques Description 

We are going to split our dataset into training, validation, and test datasets. Stick to the golden rule, we will train and tune models with the training and validation datasets only. We decided to focus on the encoder-decoder model as it's the most common method for remote sensing images captioning. Here are the three encoder-decoder models we will try:

1. Our first model will be a simple model, with a CNN layer as the encoder and an LSTM layer as the decoder. We will let the CNN layer learn images features from training images and train a word embedding with training captions, and then pass both outputs to the LSTM layer for caption generation. This baseline model should give us some sense of the bottom line.

2. The second model will have an attention structure on top of the baseline model. Unlike natural images, remote sensing images usually contain many objects without a focus and thus require more detailed captions. The CNN output is a high-level image summary and does not carry enough information for a detailed caption. Xu et al. [1] suggested using an attention model to move the focus across an image to capture all important details. As proposed by Zhang et al. [2], adding an attention structure that extracts low-level features from the CNN convolution layer could improve the model performance. We will try this architecture and would expect this model to produce more detailed captions compared to the baseline.

3. As an extension of the second model, the third model will contain three attention structures on top of the baseline model. Li et al. [3] proposed a multi-level attention model that better mimics human attention mechanisms to improve remote sensing image captioning. This multi-level attention model includes the first attention structure to focus on the image, the second attention structure to focus on semantic information, and the third attention structure to decide whether to generate the next word primarily based on vision or semantics. We are going to implement this architecture and expect this model to produce cations of the best quality.

[insert model architecture graph here]

If time permits, we could explore other model architectures and try fine-tuning pre-trained cross-modal models. To access those models, we can use some evaluation metrics suggested in this paper [3], including BLEU, Meteor, ROUGE_L, CIDEr, and SPICE. Finally, we will test our best model with the test dataset and evaluate the results (manually check?).

## Timeline and Evaluation:
 
The length of our capstone project is two months, starting from May 2020 to June 2020. During the eight weeks, we would like to achieve milestones for the following five stages: proposal, database design, model development, visualization design and polishing. The first two weeks would be used for the proposal stage, we planned to deliver both an oral and written proposal. The next four weeks would be used for data product development. We would design our database, deep learning model and visualization dashboard in parallel during this period. Both the database and dashboard design would run one week while the deep model development would run three weeks. Three milestones would be achieved by the end of this data product development stage. The last two weeks would be used to improve and polish the final product based on feedback from mentor and MDA partners. We would deliver the final presentation, final written report and final data products to both MDS mentor and MDA partners.
 
[Insert image here]
Sketch below for discussion in meeting, will draw in PPT afterwards
 

## Reference

1. K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel, and Y. Bengio. Show, attend and tell: Neural image caption generation with visual attention. International Conference on Machine Learning, 2015, 2048–2057.

2. Zhang, X.; Wang, X.; Tang, X.; Zhou, H.; Li, C. Description Generation for Remote Sensing Images Using Attribute Attention Mechanism. Remote Sens. 2019, 11, 612.

3. Li, Y.; Fang, S.; Jiao, L.; Liu, R.; Shang, R. A Multi-Level Attention Model for Remote Sensing Image Captions. Remote Sens. 2020, 12, 939. 


