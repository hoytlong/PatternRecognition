# -*- coding: utf-8 -*-

#coded in Python 2.7 

#########
#imports
#########

from __future__ import division
import nltk, re, pprint

import codecs
import sys
sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
sys.stdin = codecs.getreader('utf_8')(sys.stdin)
from nltk.corpus import cmudict
from featx import *
import numpy as np
from numpy import mean, median
from HaikuClassificationFunctions import *
import os

#load dictionary with stored syllable counts
temp_syl_dict = unpickle_syl_dict()
text_syl_dict = {}
#instantiates the carnegie mellon pronunciation dictionary
d = cmudict.dict()  

######################################################################################
#Function to run multiple classification tests on haiku and short-poem corpora
######################################################################################
#journal - assign the name of the journal you want to classify against; "Combo" allows for combining multiple journals
#trans_or_adapt - set to "trans" if classifying translated haiku corpus against the journal, else set to "adapt" (i.e., adaptations)
#count_syls - set to 1 if you want to include syllables in the classification, else set to 0
#iterations - how many times should the classification be run?
def classify_haiku(journal, trans_or_adapt, count_syls, iterations, text_syl_dict, temp_syl_dict, d):  
    #build the haiku corpus
    haiku_labeled = build_haiku(trans_or_adapt, count_syls, text_syl_dict, temp_syl_dict, d)
    #build journal corpus based on input
    if journal == "Poetry":
        start_year = raw_input("Enter the start-year for Poetry corpus: ")
        end_year = raw_input("Enter the end-year for Poetry corpus: ")
        max_length = int(raw_input("Enter the max-length of poem for Poetry corpus: "))
        short_labeled = build_short_corpus(start_year, end_year, max_length, trans_or_adapt, count_syls, text_syl_dict, temp_syl_dict, d)
    elif journal == "Combo":
        journals = []
        journals = raw_input("Please enter the journals to combine separated by just comma: ").split(",")
        max_length = raw_input("Enter the max-length of poem for journal corpus: ")
        short_labeled = []
        for item in journals:
            short_labeled += labeled_short_corpus(item, int(max_length), trans_or_adapt, count_syls)
    else:
        max_length = raw_input("Enter the max-length of poem for journal corpus: ")
        short_labeled = labeled_short_corpus(journal, int(max_length), trans_or_adapt, count_syls, text_syl_dict, temp_syl_dict, d)
    #find the smaller corpus and set corpus_length to that value
    if len(short_labeled) < len(haiku_labeled):
        corpus_length = len(short_labeled)
    else:
        corpus_length = len(haiku_labeled)
    #make sure things are working right
    print '# of poems in haiku corpus: ', len(haiku_labeled)
    print '# of poems in short corpus: ', len(short_labeled)
    #print corpus_length
        
    #perform 4-fold cross validation
    accuracy_scores, most_informative, h_precision, h_recall, nh_precision, nh_recall = cross_validate(iterations, haiku_labeled, short_labeled, corpus_length, journal) 
    #select out the most informative features
    top_non_haiku, top_haiku = get_top_features(most_informative)
    #do randomized test to check for significance of accuracy scores
    random_scores = random_test(iterations, haiku_labeled, short_labeled, corpus_length)
    #print the mean and median for normal and random tests
    print("\nAccuracy Scores for Normal Test")
    print("Mean Accuracy: {0}".format(mean(accuracy_scores)))
    print("Median Accuracy: {0}".format(median(accuracy_scores)))
    print("Median Precision for Haiku: {0}".format(median(h_precision)))
    print("Median Recall for Haiku: {0}".format(median(h_recall)))
    print("Median Precision for Not-Haiku: {0}".format(median(nh_precision)))
    print("Median Recall for Not-Haiku: {0}".format(median(nh_recall)))
    print("\nAccuracy Scores for Randomized Test")
    print("Mean Accuracy: {0}".format(mean(random_scores)))
    print("Median Accuracy: {0}".format(median(random_scores)))
    print("\nTop 20 Features for Predicting Non-Haiku:")
    print top_non_haiku
    print("\nTop 20 Features for Predicting Haiku:")
    print top_haiku