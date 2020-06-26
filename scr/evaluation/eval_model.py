# author: Dora Qian
# date: 2020-06-10

'''This script calculates evaluation scores for test results.
This script takes the path to the data folder and the caption file name and save the scores under data/score.

Usage: scr/evaluation/eval.py --root_path=<root_path> --inputs=<inputs>

Options:
--root_path=<root_path>      The path of the data folder.
--inputs=<inputs>            The name of the caption file to process.

Example:
python scr/evaluation/eval.py --root_path=data --inputs=sydney

Takes the path to the data/json folder containing the raw sydney json file and the model generated caption 
json files of sydney dataset, then calculates the evaluation scores for sydney dataset and store the score in
data/score folder。
'''

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.usc_sim.usc_sim import usc_sim
import subprocess
from docopt import docopt
import json
import os 

opt = docopt(__doc__)

def eval_model(root_path, inputs):
    """
    Computes evaluation metrics of the model results against the human annotated captions
    
    Parameters:
    ------------
    root_path: str
        the path to the data folder which contains the raw folder
    inputs: str
        the name of the caption file to process
        
    Returns:
    ------------
    None, it saves the overall score and individual score files under output path
    """
    # load data
    try :
        with open(f'{root_path}/json/{inputs}.json', 'r') as data:
            ref_data = json.load(data)
    except:
        raise(f'Make sure that human-annotated captions are store in',
        f'{root_path}/json/{inputs}.json.')
    
    try :
        with open(f'{root_path}/json/{inputs}_model_caption.json', 'r') as data:
            results = json.load(data)
    except:
        raise('Please call generate_captions.py to generate captions first.')

    # download stanford nlp library
    subprocess.call(['scr/evaluation/get_stanford_models.sh'])
    
    # format the inputs
    img_id_dict = {'image_id': list(ref_data.keys())}

    imgIds = img_id_dict['image_id']
    gts = {}
    res = {}

    required_key = { 'raw', 'imgid', 'sentid' }

    for imgId in imgIds:
        caption_list = ref_data[imgId]['sentences']
        caption_list_sel = []
        for i in caption_list:
            lst = { key:value for key,value in i.items() if key in required_key}
            lst['caption'] = lst.pop('raw')
            lst['image_id'] = lst.pop('imgid')
            lst['id'] = lst.pop('sentid')
            caption_list_sel.append(lst)
        gts[imgId] = caption_list_sel

        generated = [{'caption': results[imgId]}]
        res[imgId] = generated
        
    # tokenize
    print('tokenization...')
    tokenizer = PTBTokenizer()
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    
    # compute scores
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
        (usc_sim(), "USC_similarity"),  
        ]
    score_dict = {}
    scores_dict = {}
    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                score_dict[m] = sc
                scores_dict[m] = scs
        else:
            score_dict[method] = score
            scores_dict[method] = scores
            
    # format the individual scores
    img_score_dict = {}
    for n in range(len(res)):
        img_name = list(res.keys())[n]
        img_score_dict[img_name] = {}
        for metrics in scores_dict.keys():
            if metrics == 'SPICE':
                img_score_dict[img_name][metrics] = scores_dict[metrics][n]['All']['f']
            else:
                img_score_dict[img_name][metrics] = scores_dict[metrics][n]
    
    output_path = f'{root_path}/score'
    # save the overall score and individual image score
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    with open(f'{output_path}/{inputs}_score.json', 'w') as file:
        json.dump(score_dict, file)
        
    with open(f'{output_path}/{inputs}_img_score.json', 'w') as file:
        json.dump(img_score_dict, file)

    assert os.path.isfile(f'{output_path}/{inputs}_score.json'),\
    "Average scores are not saved."
    assert os.path.isfile(f'{output_path}/{inputs}_img_score.json'),\
    "Individual scores are not saved."

if __name__ == "__main__":
    eval_model(opt["--root_path"], opt["--inputs"])