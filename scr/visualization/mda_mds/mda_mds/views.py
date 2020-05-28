from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from subprocess import run, PIPE
import sys


def button(request):
    return render(request, 'index.html')

def output(request):
    score = 666
    data = str(score)
    print("the score is " + data)
    return render(request, 'index.html', {'data':data})


def external(request):
    user_caption_input = request.POST.get('param')
    image = request.FILES['image_upload']
    print("image is ", image)
    fs = FileSystemStorage()

    # current image name
    filename = fs.save(image.name, image)

    # absolute image path
    fileurl = fs.open(filename)

    # relative image path
    templateurl = fs.url(filename)



    print(filename)
    #output = run([sys.executable,'//Users//apple//Documents//MDS_labs//DSCI_591//591_capstone_2020-mda-mds//scr//visualization//mda_mds//mda_mds//test.py', input], shell=False, stdout=PIPE)
    image = run([sys.executable,'//Users//apple//Documents//MDS_labs//DSCI_591//591_capstone_2020-mda-mds//scr//visualization//mda_mds//mda_mds//image.py',
                            str(fileurl), str(filename), str(user_caption_input)], shell=False, stdout=PIPE, universal_newlines=True)
    output = str(image.stdout).replace('Upload Successful','').split('_')
    score = output[0]
    model_caption = output[1]
    print("IMAGE STD OUT, NEW RELATIVE PATH FOR IMAGE"+str(image.stdout).replace('Upload Successful',''))
    return render(request, 'index.html', {'data':str(image.stdout).replace('Upload Successful',''), 'raw_url':templateurl,
                            'edit_url':image.stdout, 'score':score, 'user_caption':user_caption_input, 'model_caption':model_caption})
