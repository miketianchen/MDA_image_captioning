from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.template import RequestContext

from subprocess import run, PIPE

import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def button(request):
     return render(request, 'index.html')

def output(request):
    #### TO BE COMPLETED IDK WHEN
    score = 666
    data = str(score)
    #print("the score is " + data)
    return render(request, 'index.html', {'data':data})

def generate(request):
    # path for the generate script
    generate_script_path = os.path.join(BASE_DIR, 'mda_mds/generate_image.py')
    #print("THE BASE DIR IS " + str(generate_script_path))
    out= run([sys.executable,generate_script_path],
                        shell=False, stdout=PIPE, universal_newlines=True)
    output = str(out.stdout)
    output_list = output.split("@")
    # IMAGE ONE DATA
    image_one_data = output_list[0]
    image_one_list = image_one_data.split("%")
    image_one_name = image_one_list[0]
    image_one_bleu_1 = image_one_list[1]
    image_one_meteor = image_one_list[2]
    image_one_rouge_l = image_one_list[3]
    image_one_cider = image_one_list[4]
    image_one_spice = image_one_list[5]
    image_one_usc = image_one_list[6]
    image_one_caption = image_one_list[7]
    image_one_og_captions = image_one_list[8]
    # IMAGE TWO DATA
    image_two_data = output_list[1]
    image_two_list = image_two_data.split("%")
    image_two_name = image_two_list[0]
    image_two_bleu_1 = image_two_list[1]
    image_two_meteor = image_two_list[2]
    image_two_rouge_l = image_two_list[3]
    image_two_cider = image_two_list[4]
    image_two_spice = image_two_list[5]
    image_two_usc = image_two_list[6]
    image_two_caption = image_two_list[7]
    image_two_og_captions = image_two_list[8]

    #print(str(out.stdout))
    return render(request, 'index.html', {'generated_caption_1':image_one_caption,
                'bleu_1_1':image_one_bleu_1, 'meteor_1':image_one_meteor,
                'rouge_l_1':image_one_rouge_l, 'cider_1':image_one_cider,
                'spice_1':image_one_spice, 'image_one_name':image_one_name,
                'usc_1':image_one_usc, 'og_caption_1':image_one_og_captions,
                'generated_caption_2':image_two_caption,
                'bleu_1_2':image_two_bleu_1, 'meteor_2':image_two_meteor,
                'rouge_l_2':image_two_rouge_l, 'cider_2':image_two_cider,
                'spice_2':image_two_spice, 'image_two_name':image_two_name,
                'usc_2':image_two_usc, 'og_caption_2':image_two_og_captions})

def external(request):
    if 'upload_image_input' in request.POST:
        upload_mode = "image"
        # path for the image script
        image_script_path = os.path.join(BASE_DIR, 'mda_mds/image.py')


        image = request.FILES['image_upload']
        #print("image is ", image)
        fs = FileSystemStorage()

        # current image name
        image_name = image.name
        image_name_split = image_name.split(".")
        image_name = image_name_split[0] + str(int(time.time())) + "." + image_name_split[1]

        filename = fs.save(image_name, image)

        # absolute image path
        fileurl = fs.open(filename)

        #print(os.path.dirname(str(fileurl)))

        # relative image path
        templateurl = fs.url(filename)



        #print(raw_url)
        #output = run([sys.executable,'//Users//apple//Documents//Web_dev//django-mda//mda_mds//mda_mds//test.py', input], shell=False, stdout=PIPE)
        image = run([sys.executable,image_script_path,
                                str(upload_mode), str(fileurl), str(filename)], shell=False, stdout=PIPE, universal_newlines=True)
        sys_out = str(image.stdout).replace('Upload Successful','')
        output = sys_out.split('_')
        score = output[0]
        model_caption = output[1]
        print("SYSTEM OUT IS "+sys_out)
        #print("IMAGE STD OUT, NEW RELATIVE PATH FOR IMAGE"+str(image.stdout).replace('Upload Successful',''))
        return render(request, 'index.html', {'data':str(image.stdout).replace('Upload Successful',''), 'raw_url':templateurl,
                                'edit_url':image.stdout, 'score':score, 'model_caption':model_caption})
    elif 'upload_caption_input' in request.POST:
        upload_mode = "caption"
        # path for the image script
        image_script_path = os.path.join(BASE_DIR, 'mda_mds/image.py')

        user_caption_input = request.POST.get('param')
        optional_caption_2 = request.POST.get('param_2')
        optional_caption_3 = request.POST.get('param_3')
        optional_caption_4 = request.POST.get('param_4')
        optional_caption_5 = request.POST.get('param_5')


        image = run([sys.executable,image_script_path,
                                str(upload_mode), str(user_caption_input), str(optional_caption_2), str(optional_caption_3), str(optional_caption_4), str(optional_caption_5)]
                                , shell=False, stdout=PIPE, universal_newlines=True)
        return render(request, 'index.html')


    # # path for the image script
    # image_script_path = os.path.join(BASE_DIR, 'mda_mds/image.py')
    #
    # user_caption_input = request.POST.get('param')
    # optional_caption_2 = request.POST.get('param_2')
    # optional_caption_3 = request.POST.get('param_3')
    # optional_caption_4 = request.POST.get('param_4')
    # optional_caption_5 = request.POST.get('param_5')
    #
    # image = request.FILES['image_upload']
    # #print("image is ", image)
    # fs = FileSystemStorage()
    #
    # # current image name
    # filename = fs.save(image.name, image)
    #
    # # absolute image path
    # fileurl = fs.open(filename)
    #
    # #print(os.path.dirname(str(fileurl)))
    #
    # # relative image path
    # templateurl = fs.url(filename)
    #
    #
    #
    # #print(raw_url)
    # #output = run([sys.executable,'//Users//apple//Documents//Web_dev//django-mda//mda_mds//mda_mds//test.py', input], shell=False, stdout=PIPE)
    # image = run([sys.executable,image_script_path,
    #                         str(fileurl), str(filename), str(user_caption_input), str(optional_caption_2), str(optional_caption_3), str(optional_caption_4), str(optional_caption_5)]
    #                         , shell=False, stdout=PIPE, universal_newlines=True)
    # sys_out = str(image.stdout).replace('Upload Successful','')
    # output = sys_out.split('_')
    # score = output[0]
    # model_caption = output[1]
    # print("SYSTEM OUT IS "+sys_out)
    # #print("IMAGE STD OUT, NEW RELATIVE PATH FOR IMAGE"+str(image.stdout).replace('Upload Successful',''))
    # return render(request, 'index.html', {'data':str(image.stdout).replace('Upload Successful',''), 'raw_url':templateurl,
    #                         'edit_url':image.stdout, 'score':score, 'user_caption':user_caption_input, 'model_caption':model_caption})

def database(request):
    # PATH to LOCAL DIRECTORY which temporary houses the images the user uploaded
    image_temp_save_dir = os.path.join(BASE_DIR, 'media/database_images')
    # PATH to LOCAL DIRECTORY which temporary houses the json captions the user uploaded
    json_temp_save_dir = os.path.join(BASE_DIR, 'media/caption_json')

    # PATH to the SCRIPT which will upload the images to AWS S3 and then
    #       DELETE the images from local directory
    database_upload_script_path = os.path.join(BASE_DIR, 'mda_mds/database_upload_script.py')

    # def handle_uploaded_file(f):
    #     with open(image_temp_save_dir + '/file_' + str(count) + '.jpg', 'wb+') as destination:
    #         for chunk in f.chunks():
    #             destination.write(chunk)
    #
    # def handle_json_file(f):
    #     with open(json_temp_save_dir + '/captions_' + str(count) + '.json', 'wb+') as destination:
    #         for chunk in f.chunks():
    #             destination.write(chunk)

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
    # return HttpResponse("Files Successfully Uploaded!")

    out = run([sys.executable,database_upload_script_path],
                        shell=False, stdout=PIPE, universal_newlines=True)

    # print(out.stdout)

    return render(request, 'index.html')
