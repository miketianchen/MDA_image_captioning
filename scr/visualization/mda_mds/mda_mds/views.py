"""
The central script which controls the callbacks of each button/tabs in the visualization

This file will control and call other scripts in order to carry out the desired
functionality

In this script, relevant parameters will be passed to each of the other scripts
and will recieve relevant parameters from those scripts to display on the app
"""

from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.template import RequestContext

from subprocess import run, PIPE

import sys
import os
import time
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'data')

######################## HELPER FUNCTION, NOT RELATED TO VIEWS ################
def get_model_list():
    """
    This function is for the dropdown option in "User Image" tab

    This function will get the name of all the saved local models to display
    to the user in the selectable dropdown
    """
    models = list()
    for filename in os.listdir(os.path.join(DATA_PATH, 'results')):
        if filename.endswith(".hdf5"):

            # display in dropdown without .hdf5 extenstion
            models.append(filename.split(".")[0])

            # display in dropdown with .hdf5 extenstion
            # models.append(filename)

            continue
        else:
            continue
    return models
###############################################################################
model_list = get_model_list()

def index(request):
    """
    For the url route '/' baseurl (homepage)
    """
    # test_list = ['final_model', 'test_model', 'base_model', 'doggy']
    return render(request, 'index.html', {'model_list':model_list})

# TEST
def output(request):
    # TEST
    score = 0
    data = str(score)
    return render(request, 'index.html', {'data':data})

def generate(request):
    """
    For the url route '/generate/'

    This is for the "Demo Example" tab.

    This will be called when the user presses the 'Generate' button in the "Demo Example" tab

    This will call the 'mda_mds/generate_image.py' script
        - Send : Nothing
        - Recieve : Will recieve
                        * the two image names
                        * evaluation metrics for both images
                        * 5 original captions for both images
                        * model generated captions for both images

                            All of which are in a single string.
                            '@' delimits the two images
                            '%' delimits each individual data

    """

    # path for the generate script
    generate_script_path = os.path.join(BASE_DIR, 'mda_mds/generate_image.py')
    out= run([sys.executable,generate_script_path],
                        shell=False, stdout=PIPE, universal_newlines=True)
    output = str(out.stdout)
    output_list = output.split("@")
    # IMAGE ONE DATA
    image_one_data = output_list[0]
    image_one_list = image_one_data.split("%")
    image_one_generated_cap = image_one_list[0]
    image_one_input_sen_1 = image_one_list[1]
    image_one_input_sen_2 = image_one_list[2]
    image_one_input_sen_3 = image_one_list[3]
    image_one_input_sen_4 = image_one_list[4]
    image_one_input_sen_5 = image_one_list[5]
    image_one_bleu_1 = image_one_list[6]
    image_one_bleu_2 = image_one_list[7]
    image_one_bleu_3 = image_one_list[8]
    image_one_bleu_4 = image_one_list[9]
    image_one_meteor = image_one_list[10]
    image_one_rouge_l = image_one_list[11]
    image_one_cider = image_one_list[12]
    image_one_spice = image_one_list[13]
    image_one_usc = image_one_list[14]
    image_one_name = image_one_list[15]
    # IMAGE TWO DATA
    image_two_data = output_list[1]
    image_two_list = image_two_data.split("%")
    image_two_generated_cap = image_two_list[15]
    image_two_input_sen_1 = image_two_list[1]
    image_two_input_sen_2 = image_two_list[2]
    image_two_input_sen_3 = image_two_list[3]
    image_two_input_sen_4 = image_two_list[4]
    image_two_input_sen_5 = image_two_list[5]
    image_two_bleu_1 = image_two_list[6]
    image_two_bleu_2 = image_two_list[7]
    image_two_bleu_3 = image_two_list[8]
    image_two_bleu_4 = image_two_list[9]
    image_two_meteor = image_two_list[10]
    image_two_rouge_l = image_two_list[11]
    image_two_cider = image_two_list[12]
    image_two_spice = image_two_list[13]
    image_two_usc = image_two_list[14]
    image_two_name = image_two_list[0]

    #print(str(out.stdout))
    return render(request, 'index.html', {'og_caption_1_1':image_one_input_sen_1,
                'og_caption_1_2':image_one_input_sen_2, 'og_caption_1_3':image_one_input_sen_3,
                'og_caption_1_4':image_one_input_sen_4, 'og_caption_1_5':image_one_input_sen_5,
                'generated_caption_1':image_one_generated_cap, 'image_one_name':image_one_name,
                'bleu_1_1':image_one_bleu_1, 'bleu_2_1':image_one_bleu_2,
                'bleu_3_1':image_one_bleu_3, 'bleu_4_1':image_one_bleu_4,
                'rouge_l_1':image_one_rouge_l, 'cider_1':image_one_cider,
                'spice_1':image_one_spice, 'usc_1':image_one_usc,
                'meteor_1':image_one_meteor, 'meteor_2':image_two_meteor,
                'og_caption_2_1':image_two_input_sen_1, 'og_caption_2_2':image_two_input_sen_2,
                'og_caption_2_3':image_two_input_sen_3, 'og_caption_2_4':image_two_input_sen_4,
                'og_caption_2_5':image_two_input_sen_5, 'generated_caption_2':image_two_generated_cap,
                'image_two_name':image_two_name, 'bleu_1_2':image_two_bleu_1,
                'bleu_2_2':image_two_bleu_2, 'bleu_3_2':image_two_bleu_3,
                'bleu_4_2':image_two_bleu_4, 'rouge_l_2':image_two_rouge_l,
                'cider_2':image_two_cider,
                'spice_2':image_two_spice, 'usc_2':image_two_usc,'active_tab':'demo_tab', 'model_list':model_list})

def external(request):
    """
    For the url route '/external/'

    This is for the "User Image" tab.

    This tab has two 'upload_mode's for two functionalities.
    The first 'upload_mode' is "image", this is when the user first uploads the image for model captioning
    The second 'upload_mode' is "caption", this is for active learning, when the user wants to submit captions for the image

    The image the user uplaods is in saved temporary in ../visualization/mda_mds/media/ until the user presses 'Submit Caption'
    which is when the image will be moved to the appropriate folder in raw/upload
    """

    if 'upload_image_input' in request.POST:
        upload_mode = "image"
        # path for the image script
        image_script_path = os.path.join(BASE_DIR, 'mda_mds/image.py')

        selected_model = request.POST['local_ml_models']

        image = request.FILES['image_upload']
     
        if not os.path.exists(f'{DATA_PATH}/raw/upload'):
            os.makedirs(f'{DATA_PATH}/raw/upload', exist_ok=True)     
        
        im = Image.open(image).resize((299, 299), Image.ANTIALIAS)
        filename, extension = os.path.splitext(image.name.lower())
        filename = f'{filename}{str(int(time.time()))}.jpg'
        fileurl = f'{DATA_PATH}/raw/upload/{filename}'

        if extension != '.jpg':
            rgb_im = im.convert('RGB')
            rgb_im.save(fileurl, 'JPEG', quality = 95)
        else:
            im.save(fileurl, quality = 95)

        image = run([sys.executable,
                     image_script_path,
                     str(upload_mode), 
                     str(fileurl), 
                     str(filename),                    
                     str(selected_model)], 
                    shell=False, 
                    stdout=PIPE, 
                    universal_newlines=True)
        
        sys_out = str(image.stdout).replace('Upload Successful','')
        output = sys_out.split('*')
        score = output[0]
        model_caption = output[1]
        print("SYSTEM OUT IS "+ selected_model)
        return render(request, 'index.html', 
                      {'data':str(image.stdout).replace('Upload Successful',''), 
                       'raw_url':fileurl,
                       'edit_url':image.stdout,
                       'score':score, 
                       'model_caption':model_caption, 
                       'model_list':model_list})
    
    elif 'upload_caption_input' in request.POST:
        upload_mode = "caption"
        # path for the image script
        image_script_path = os.path.join(BASE_DIR, 'mda_mds/image.py')

        user_caption_input = request.POST.get('param')
        optional_caption_2 = request.POST.get('param_2')
        optional_caption_3 = request.POST.get('param_3')
        optional_caption_4 = request.POST.get('param_4')
        optional_caption_5 = request.POST.get('param_5')


        image = run([sys.executable,
                     image_script_path,
                     str(upload_mode), 
                     str(user_caption_input), 
                     str(optional_caption_2), 
                     str(optional_caption_3), 
                     str(optional_caption_4), 
                     str(optional_caption_5)], 
                    shell=False,
                    stdout=PIPE,
                    universal_newlines=True)
        return render(request, 'index.html', {'model_list':model_list})


def database(request):
    """
    For the url route '/database/'

    This is for the "Database Upload" tab.

    The user can upload multiple image files, they're supposed to only upload one json file. No security measures
    are implemented to ensure the uploads. 
    """
    # PATH to LOCAL DIRECTORY which temporary houses the images the user uploaded
    image_temp_save_dir = os.path.join(BASE_DIR, 'media/database_images')

    if not os.path.exists(image_temp_save_dir):
        os.makedirs(image_temp_save_dir, exist_ok=True)

    # PATH to LOCAL DIRECTORY which temporary houses the json captions the user uploaded
    json_temp_save_dir = os.path.join(BASE_DIR, 'media/caption_json')

    # PATH to the SCRIPT which will upload the images to AWS S3 and then
    #       DELETE the images from local directory
    database_upload_script_path = os.path.join(BASE_DIR, 'mda_mds/database_upload_script.py')


    def handle_file(f, type):
        # type = '.json' or = '.jpg'
        with open(image_temp_save_dir + '/uploaded_file_' + str(count) + type, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)

    for count, x in enumerate(request.FILES.getlist("multiple_images_upload")):
        if str(x).endswith('.json'):
            handle_file(x, '.json')
        elif str(x).endswith('.jpg'):
            handle_file(x, '.jpg')
        else:
            print("WRONG FILE TYPE, CHECK YOUR UPLOAD")

    out = run([sys.executable,database_upload_script_path],
                        shell=False, stdout=PIPE, universal_newlines=True)


    return render(request, 'index.html', {'model_list':model_list})
