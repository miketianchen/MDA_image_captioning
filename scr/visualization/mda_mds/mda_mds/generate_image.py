'''
Script part of the visualization module, not intended to be run separately

Script for 'Demo Example' tab's functionality
    The route (url) is /generate

This script will be called from `views.py`'s `generate` function

This script is used to grab two random images from the test set, along with the
evaluation metric scores, model generated captions and 5 original training
captions
'''

import sys, json
import random
import os


# /Users/apple/Documents/MDS_labs/DSCI_591/591_capstone_2020-mda-mds/scr/visualization/mda_mds
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PATH TO DATA FOLDER
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'data')

# PATH TO ALL THE IMAGES IN THE TEST FOLDER
IMAGES_PATH = os.path.join(DATA_PATH, 'test/')

# PATH TO THE EVALUATION SCORES FILE FOR THE TEST IMAGES
TEST_IMG_SCORE_JSON = os.path.join(DATA_PATH, 'score/test_img_score.json')

# PATH TO THE INPUT CAPTIONS FOR ALL THE IMAGES IN TEST FOLDER
TEST_CAPTIONS_JSON = os.path.join(DATA_PATH, 'json/test.json')

# PATH TO GENERATED CAPTIONS FOR ALL THE IMAGES IN TEST FOLDER
GENERATED_CAPTIONS_JSON = os.path.join(DATA_PATH, 'json/test_model_caption.json')

with open(TEST_CAPTIONS_JSON) as json_file:
    captions = json.load(json_file)

with open(TEST_IMG_SCORE_JSON) as json_file:
    evaluations = json.load(json_file)

with open(GENERATED_CAPTIONS_JSON) as json_file:
    generated_captions = json.load(json_file)

demo_image_names = os.listdir(IMAGES_PATH)
def return_captions():
    """
    Will grab two random images with captions, evaluation scores from test set

    NOTE: IF TIME PERMITS CHANGE TO AVOID REPEATING CODE!!!
    """
    # Grab random index for the image to be displayed
    random_image_one_index = random.randint(0, len(demo_image_names)-1)
    random_image_two_index = random.randint(0, len(demo_image_names)-1)

    # Grab the name of the random index of the image
    random_image_one_name = demo_image_names[random_image_one_index]
    random_image_two_name = demo_image_names[random_image_two_index]

    # Grab the 5 original training captions of the random image as a
    #   multilayered dictionary
    image_one_input_captions = captions[random_image_one_name]
    image_two_input_captions = captions[random_image_two_name]

    # Traverse through dictionary to grab each of the 5 individual trianing caption
    image_one_input_sen_1 = image_one_input_captions['sentences'][0]['raw']
    image_one_input_sen_2 = image_one_input_captions['sentences'][1]['raw']
    image_one_input_sen_3 = image_one_input_captions['sentences'][2]['raw']
    image_one_input_sen_4 = image_one_input_captions['sentences'][3]['raw']
    image_one_input_sen_5 = image_one_input_captions['sentences'][4]['raw']

    image_two_input_sen_1 = image_two_input_captions['sentences'][0]['raw']
    image_two_input_sen_2 = image_two_input_captions['sentences'][1]['raw']
    image_two_input_sen_3 = image_two_input_captions['sentences'][2]['raw']
    image_two_input_sen_4 = image_two_input_captions['sentences'][3]['raw']
    image_two_input_sen_5 = image_two_input_captions['sentences'][4]['raw']

    # Grab the Evaluation Scores from the EVALUATION SCORES File
    image_one_scores = evaluations[random_image_one_name]
    image_two_scores = evaluations[random_image_two_name]

    # Extract each of the individual metric scores
    image_one_bleu_1 = str(round(image_one_scores['Bleu_1'], 2))
    image_one_bleu_2 = str(round(image_one_scores['Bleu_2'], 2))
    image_one_bleu_3 = str(round(image_one_scores['Bleu_3'], 2))
    image_one_bleu_4 = str(round(image_one_scores['Bleu_4'], 2))
    image_one_meteor = str(round(image_one_scores['METEOR'], 2))
    image_one_rouge_l = str(round(image_one_scores['ROUGE_L'], 2))
    image_one_cider = str(round(image_one_scores['CIDEr'], 2))
    image_one_spice = str(round(image_one_scores['SPICE'], 2))
    image_one_usc = str(round(image_one_scores['USC_similarity'], 2))

    image_two_bleu_1 = str(round(image_two_scores['Bleu_1'], 2))
    image_two_bleu_2 = str(round(image_two_scores['Bleu_2'], 2))
    image_two_bleu_3 = str(round(image_two_scores['Bleu_3'], 2))
    image_two_bleu_4 = str(round(image_two_scores['Bleu_4'], 2))
    image_two_meteor = str(round(image_two_scores['METEOR'], 2))
    image_two_rouge_l = str(round(image_two_scores['ROUGE_L'], 2))
    image_two_cider = str(round(image_two_scores['CIDEr'], 2))
    image_two_spice = str(round(image_two_scores['SPICE'], 2))
    image_two_usc = str(round(image_two_scores['USC_similarity'], 2))

    # Grab the caption generated from the model
    image_one_model_generated_sentence = generated_captions[random_image_one_name]
    image_two_model_generated_sentence = generated_captions[random_image_two_name]

    # Put all of the variables to be sent back to `view.py` into a list which Will
    #   then be turned into a string
    image_one_output_list = [image_one_model_generated_sentence,
                       image_one_input_sen_1,
                       image_one_input_sen_2,
                       image_one_input_sen_3,
                       image_one_input_sen_4,
                       image_one_input_sen_5,
                       image_one_bleu_1,
                       image_one_bleu_2,
                       image_one_bleu_3,
                       image_one_bleu_4,
                       image_one_meteor,
                       image_one_rouge_l,
                       image_one_cider,
                       image_one_spice,
                       image_one_usc,
                       random_image_one_name]

    image_two_output_list = [random_image_two_name,
                       image_two_input_sen_1,
                       image_two_input_sen_2,
                       image_two_input_sen_3,
                       image_two_input_sen_4,
                       image_two_input_sen_5,
                       image_two_bleu_1,
                       image_two_bleu_2,
                       image_two_bleu_3,
                       image_two_bleu_4,
                       image_two_meteor,
                       image_two_rouge_l,
                       image_two_cider,
                       image_two_spice,
                       image_two_usc,
                       image_two_model_generated_sentence]

    # delimit the two ouput strings with %
    image_one_output_string = "%".join(image_one_output_list)
    image_two_output_string = "%".join(image_two_output_list)

    # delimit these two output with @
    print(image_one_output_string + "@" + image_two_output_string)



# def get_captions_n_scores(image_name):
#     input_captions = captions[image_name]
#     input_sentence_1 = input_captions['sentences'][0]['raw']
#     input_sentence_2 = input_captions['sentences'][1]['raw']
#     input_sentence_3 = input_captions['sentences'][2]['raw']
#     input_sentence_4 = input_captions['sentences'][3]['raw']
#     input_sentence_5 = input_captions['sentences'][4]['raw']
#
#     caption_scores = evaluations[image_name]
#
#     generated_cap = generated_captions[image_name]
#
#     output_bleu_1 = str(round(caption_scores['Bleu_1'], 2))
#     output_bleu_2 = str(round(caption_scores['Bleu_2'], 2))
#     output_bleu_3 = str(round(caption_scores['Bleu_3'], 2))
#     output_bleu_4 = str(round(caption_scores['Bleu_4'], 2))
#     output_METEOR = str(round(caption_scores['METEOR'], 2))
#     output_ROUGE_L = str(round(caption_scores['ROUGE_L'], 2))
#     output_CIDEr = str(round(caption_scores['CIDEr'], 2))
#     output_SPICE = str(round(caption_scores['SPICE'], 2))
#     output_USC = str(round(caption_scores['USC_similarity'], 2))
#
#     output_list = [generated_cap,
#                    input_sentence_1,
#                    input_sentence_2,
#                    input_sentence_3,
#                    input_sentence_4,
#                    input_sentence_5,
#                    output_bleu_1,
#                    output_bleu_2,
#                    output_bleu_3,
#                    output_bleu_4,
#                    output_METEOR,
#                    output_ROUGE_L,
#                    output_CIDEr,
#                    output_SPICE,
#                    output_USC,
#                    image_name]
#     output_string = "%".join(output_list)
#
#     return output_string


return_captions()
