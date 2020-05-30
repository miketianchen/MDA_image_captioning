#!/usr/bin/env python
# 
# File Name : usc_sim.py
#
# Description : Computes Universal Sentence Encoder Similarity score
#
# Creation Date : 2020-05-29
# Author : Dora Qian
# This code is used for UBC MDS-MDA Capstone project

import numpy as np
import tensorflow_hub as hub
import tensorflow_text

class usc_sim():
    '''
    Class for computing Universal Sentence Encoder Similarity score for a set of candidate sentences 

    '''
    def __init__(self):
        self.params = 'Universal_Sentence_Encoder_Similarity'
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

    def calc_score(self, candidate, refs):
        """
        Compute Universal Sentence Encoder Similarity score given one candidate and references for an image
        
        Parameters:
        ----------
        candidate: str
            the generated sentences to be evaluated
        refs: list
            a list of reference sentences
        
        Returns:
        ---------
        float, average score of similarity between candidate and reference sentences
        """
        similarity_list = []
        embed = self.embed
        candidate_embed = embed(candidate)
        for caption in refs:
            ref_embed = embed(caption)
            similarity_score = np.inner(ref_embed, candidate_embed)
            similarity_list.append(similarity_score)
            
        score = np.mean(similarity_list, dtype = "float64")
        return score

    def compute_score(self, gts, res):
        """
        Computes Universal Sentence Encoder Similarity score given a set of reference and candidate sentences 
        
        Parameters:
        ----------
        candidate: dict
            candidate / test sentences with "image name" key and "tokenized sentences" as values
        refs: dict
            reference sentences with "image name" key and "tokenized sentences" as values
        
        Returns:
        ---------
        float, average score of similarity between candidate and reference sentences for all image in thdataset
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for i in imgIds:
            hypo = res[i][0]
            ref  = gts[i]

            score.append(self.calc_score(hypo, ref))

        average_score = np.mean(np.array(score))
        return average_score, score

    def method(self):
        return self.params