# author: Dora Qian
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.usc_sim.usc_sim import usc_sim
import subprocess


def eval_model(ref_data, results):
    """
    Computes evaluation metrics of the model results against the human annotated captions
    
    Parameters:
    ------------
    ref_data: dict
        a dictionary containing human annotated captions, with image name as key and a list of human annotated captions as values
    
    results: dict
        a dictionary containing model generated caption, with image name as key and a generated caption as value
        
    Returns:
    ------------
    score_dict: a dictionary containing the overall average score for the model
    img_score_dict: a dictionary containing the individual scores for images
    scores_dict: a dictionary containing the scores by metric type
    """
    # download stanford nlp library
    subprocess.call(['../../scr/evaluation/get_stanford_models.sh'])
    
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
                
    return score_dict, img_score_dict, scores_dict