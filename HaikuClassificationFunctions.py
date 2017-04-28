#imports

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
import os
from nltk.corpus import stopwords
import pandas as pd
import random
import collections
from nltk.classify import NaiveBayesClassifier  #import directly from .py file while in appropriate directory (then use updated function, which returns features as list)
from nltk.classify.util import accuracy    

#################################################################################################
#function to clean text files from "Poetry Magazine"
#consider not cleaning the "!" or "-", since these are highly informative in the case of haiku
#exclamation - pass a 1 or 0 to the function, depending on whether you want to include "!" or not
#################################################################################################
def clean_poetry(raw, exclamation):
    import re
    raw = re.sub(r'I\.', '', raw)
    raw = re.sub(r'\]', '', raw)
    raw = re.sub(r'[0-9]', '', raw)
    raw = re.sub(r'II', '', raw)
    if exclamation == 1:
        raw = re.sub(r'[,.;:"?*()]', '', raw)    #took out "!" [make sure to remove "!" when doing machine learning; keep in for syllable counting] 
    else:
        raw = re.sub(r'[,.;:"?!*()]', '', raw)
    raw = re.sub(r'-', ' ', raw)
    raw = raw.replace('\'s', '')
    raw = raw.replace('\'S', '')
    raw = re.sub(r'\[', '', raw)
    raw = re.sub(u'\u2014','',raw)
    raw = re.sub(u'\ufeff','',raw)
    raw = re.sub(u'\u2019','',raw)
    raw = re.sub(u'\ufb02','',raw)
    return raw

#################################
# Functions to count syllables
#################################
def unpickle_syl_dict():
    import pickle
    filename = 'c:\Users\Public\Documents\MyData\HaikuArticle\save.p'
    temp_syl_dict = pickle.load(open(filename, "rb"))   #stored in your HaikuArticle Folder
    return temp_syl_dict

def pickle_syl_dict(temp_syl_dict):
    import pickle
    filename = 'c:\Users\Public\Documents\MyData\HaikuArticle\save.p'
    pickle.dump(temp_syl_dict, open(filename, "wb"))
    
def nsyl(word, d):
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]

def count_syl(text_words, file_num, text_syl_dict, temp_syl_dict, d):   #a function to count syllables in a tokenized string
    if file_num in text_syl_dict.keys():
        return text_syl_dict[file_num]
    else:
        poem_syl = 0
        for word in text_words:
            if not word in d.keys():                        #check to see if word is not in Pronunciation dictionary
                if word in temp_syl_dict.keys():            #check to see if word is in temporary syllable count dictionary
                    poem_syl += int(temp_syl_dict[word][0]) #add the syllable count from the temporary dictionary
                else:                                       #ask user to input the number of syllables
                    print word
                    s = raw_input("How many syllables?: ")
                    poem_syl += int(s)
                    temp_syl_dict[word] = [int(s)]          #store the value in temporary syllable count dictionary so it only needs to be counted once
            else:
                syl = nsyl(word, d)[0]
                poem_syl += syl
                temp_syl_dict[word] = syl  
                #poem_syl += nsyl(word)[0]  #iterates over each word and adds syllable count to total (nsyl returns a list, thus the square brackets)
        text_syl_dict[file_num] = poem_syl
        return poem_syl

##################################################
# Functions for building corpora and feature sets
##################################################

#function to build haiku feature-set
#To build translation corpus, set trans_or_not to "trans"; otherwise set to "adapt"
#To add syllables as a feature, set count_syls to 1, otherwise 0
#the haiku metadata file lists an "x" for every poem that you do not want included in the labeled corpus
def build_haiku(trans_or_adapt, count_syls, text_syl_dict, temp_syl_dict, d):
    csv = pd.read_csv('c:\Users\Public\Documents\MyData\HaikuArticle\Haikus.csv')
    haiku_labeled = []
    directory = 'c:\Users\Public\Documents\MyData\HaikuArticle\Haikus\\' #point to directory of known "haikus"
    lemma = raw_input("Lemmatize? y/n?")
    for file in os.listdir(directory):
        file_num = re.sub(r'.txt', '', file)
        if trans_or_adapt == "trans":
            if csv[csv.filename==file].journal.values == 'trans' and csv[csv.filename==file].type.values != 'x':
                if count_syls == 1:
                    text = codecs.open(directory+file, "r", "utf-8")
                    raw = text.read()
                    raw = clean_poetry(raw, 0)
                    text_words = nltk.word_tokenize(raw.lower())
                    poem_syl = count_syl(text_words, file_num, text_syl_dict, temp_syl_dict, d)            #get syllable count of poem
                text = codecs.open(directory+file, "r", "utf-8")
                raw = text.read()                           #refresh the raw text file to get a clean copy
                raw = clean_poetry(raw, 1)                  #clean again, but this time do not exclude "!"    
                text_words = nltk.word_tokenize(raw.lower())
                if lemma == "y":
                    wnl = nltk.WordNetLemmatizer()
                    text_words = [wnl.lemmatize(w) for w in text_words] 
                feature = bag_of_non_stopwords(text_words, stopfile='english')  #grab all words, not including stopwords
                                                                                    #rev_english excludes "you", "your", "our"
                if count_syls == 1:
                    if poem_syl <= 18:                          #use <= 20 for pre-1914 corpus
                        feature["less_than_18_syl"] = True
                    if poem_syl > 18:
                        feature["less_than_18_syl"] = False
                label = 'haiku'
                haiku_labeled.append([file, [feature, label]])
        else:        
            if csv[csv.filename==file].journal.values == 'adapt' and csv[csv.filename==file].type.values != 'x':
                if count_syls == 1:
                    text = codecs.open(directory+file, "r", "utf-8")
                    raw = text.read()
                    raw = clean_poetry(raw, 0)
                    text_words = nltk.word_tokenize(raw.lower())
                    poem_syl = count_syl(text_words, file_num, text_syl_dict, temp_syl_dict, d)            #get syllable count of poem
                text = codecs.open(directory+file, "r", "utf-8")
                raw = text.read()                           #refresh the raw text file to get a clean copy
                raw = clean_poetry(raw, 1)                  #clean again, but this time do not exclude "!"    
                text_words = nltk.word_tokenize(raw.lower())
                if lemma == "y":
                    wnl = nltk.WordNetLemmatizer()
                    text_words = [wnl.lemmatize(w) for w in text_words] 
                feature = bag_of_non_stopwords(text_words, stopfile='english')  #grab all words, not including stopwords
                if count_syls == 1:
                    if poem_syl <= 19:                          #use <= 19 for adaptation corpus
                        feature["less_than_19_syl"] = True
                    if poem_syl > 19:
                        feature["less_than_19_syl"] = False
                    if poem_syl > 19 and poem_syl <= 31:       #this captures a second bin; only use for adaptations
                        feature["btw_19_and_31"] = True
                    if poem_syl < 19 and poem_syl > 31:
                        feature["btw_19_and_31"] = False
                #if len(raw) < 250:
                #feature["less_than_250"] = True
                #if (len(raw) > 250):
                #feature["less_than_250"] = False
                label = 'haiku'
                haiku_labeled.append([file, [feature, label]])
    return haiku_labeled 

#this is used to excise hokku in our own corpus from the Poetry magazine files
def is_hokku(file_num):
    hokku_poems = ['20571115','20571116','20571117','20571118','20571119','20571120','20571121','20571122','20571123','20571124','20571125','20571126','20571127','20571128','20571129','20571130','20571131','20574100','20574101','20574102','20570962','20572348','20572152','20570658','20569747','20569927','20569928']
    if file_num in hokku_poems:
        return True                           

def is_hokku_Others(file_num):
    hokku_poems = ['oth0141','oth0142','oth0220','oth0221','oth0414','oth0151','oth0139','oth0150']
    if file_num in hokku_poems:
        return True                           
	
#function to create labeled feature-set for Poetry corpus
#give the function start and end years; no earlier than 1912 and no later than 1923
def build_short_corpus(start_year, end_year, max_length, trans_or_adapt, count_syls, text_syl_dict, temp_syl_dict, d):
    #build dictionary that associates years with the file-name that marks last poem of each year
    poem_years = {'1911':'0', '1912':'20569677', '1913':'20569950', '1914':'20570246', '1915':'20570653', '1916':'20571019', '1917':'20571506', '1918':'20571951', 
    '1919':'20572410', '1920':'20572900', '1921':'20573386', '1922':'20573879', '1923':'20574338'}

    short_labeled = []
    directory = 'c:\Users\Public\Documents\MyData\Poets\USData\PoetryTexts\\' #point to where your short poems are stored
    lemma = raw_input("Lemmatize? y/n?")
    for file in os.listdir(directory):
        file_num = re.sub(r'.txt', '', file) 
        if is_hokku(file_num)==None:
            if int(file_num) <= int(poem_years[end_year]) and int(file_num) > int(poem_years[str((int(start_year) - 1))]):  
                text = codecs.open(directory+file, "r", "utf-8")
                raw = text.read()
                if len(raw) <= max_length:
                    if count_syls == 1:
                        raw = clean_poetry(raw, 0)
                        text_words = nltk.word_tokenize(raw.lower())
                        poem_syl = count_syl(text_words, file_num, text_syl_dict, temp_syl_dict, d)              #get syllable count of poem
                    text = codecs.open(directory+file, "r", "utf-8")  #refresh the raw text file to get a clean copy
                    raw = text.read()                           
                    raw = clean_poetry(raw, 1)                  #clean again, but this time do not exclude "!"    
                    text_words = nltk.word_tokenize(raw.lower())
                    if lemma == "y":
                        wnl = nltk.WordNetLemmatizer()
                        text_words = [wnl.lemmatize(w) for w in text_words] 
                    feature = bag_of_non_stopwords(text_words, stopfile='english')
                    if count_syls == 1:
                        if trans_or_adapt == "trans":
                            if poem_syl <= 18:                          #use <= 20 for pre-1914 corpus; use <= 22 for post-1914 corpus
                                feature["less_than_18_syl"] = True
                            if poem_syl > 18:
                                feature["less_than_18_syl"] = False
                        else:
                            if poem_syl <= 19:                          #use <= 20 for pre-1914 corpus; use <= 22 for post-1914 corpus
                                feature["less_than_19_syl"] = True
                            if poem_syl > 19:
                                feature["less_than_19_syl"] = False
                            if poem_syl > 19 and poem_syl <= 31:       #this captures a second bin; only use for post-1914 hokku
                                feature["btw_19_and_31"] = True
                            if poem_syl < 19 and poem_syl > 31:
                                feature["btw_19_and_31"] = False
                    #if len(raw) < 250:
                    #feature["less_than_250"] = True
                    #if (len(raw) > 250):
                    #feature["less_than_250"] = False
                    label = 'not-haiku'
                    short_labeled.append([file, [feature, label]])  
    return short_labeled    
	
#function to build labeled corpus of poems under a specified length; use for any journals other than Poetry Magazine (see above)
def labeled_short_corpus(journal, max_length, trans_or_adapt, count_syls, text_syl_dict, temp_syl_dict, d):
    import os
    from nltk.corpus import stopwords
    short_labeled = []
    directory = 'c:\Users\Public\Documents\MyData\Poets\USData\JournalCorpora/' + journal + '\\' #point to where your poems are stored
    lemma = raw_input("Lemmatize? y/n?")
    for file in os.listdir(directory):
        file_num = re.sub(r'.txt', '', file)   
        if is_hokku_Others(file_num)==None:    #check for duplicates here and remove (only duplicates found so far are in Others)
            text = codecs.open(directory+file, "r", "utf-8")
            raw = text.read()
            if len(raw) <= max_length:
                if count_syls == 1:
                    raw = clean_poetry(raw, 0)
                    text_words = nltk.word_tokenize(raw.lower())
                    poem_syl = count_syl(text_words, file_num, text_syl_dict, temp_syl_dict, d)            #get syllable count of poem
                text = codecs.open(directory+file, "r", "utf-8")
                raw = text.read()                           #refresh the raw text file to get a clean copy
                raw = clean_poetry(raw, 1)                  #clean again, but this time do not exclude "!"    
                text_words = nltk.word_tokenize(raw.lower())
                if lemma == "y":
                    wnl = nltk.WordNetLemmatizer()
                    text_words = [wnl.lemmatize(w) for w in text_words] 
                feature = bag_of_non_stopwords(text_words, stopfile='english')
                if count_syls == 1:
                    if trans_or_adapt == "trans":
                        if poem_syl <= 18:                          #use <= 20 for pre-1914 corpus; use <= 22 for post-1914 corpus
                            feature["less_than_18_syl"] = True
                        if poem_syl > 18:
                            feature["less_than_18_syl"] = False
                    else:
                        if poem_syl <= 19:                          #use <= 20 for pre-1914 corpus; use <= 22 for post-1914 corpus
                            feature["less_than_19_syl"] = True
                        if poem_syl > 19:
                            feature["less_than_19_syl"] = False
                        if poem_syl > 19 and poem_syl <= 31:       #this captures a second bin; only use for post-1914 hokku
                            feature["btw_19_and_31"] = True
                        if poem_syl < 19 and poem_syl > 31:
                            feature["btw_19_and_31"] = False 
                label = 'not-haiku'
                short_labeled.append([file, [feature, label]])  
    return short_labeled    

########################################
# Functions for extracting text lengths	
########################################
	
#for Poetry Magazine
def get_poetry_lengths(year):
    #build a dictionary of all poems in the Poetry corpus and attaches their char length
	#set boundaries for years based on filename
	poem_years = {'1911':'0', '1912':'20569677', '1913':'20569950', '1914':'20570246', '1915':'20570653', '1916':'20571019', '1917':'20571506', '1918':'20571951', 
	'1919':'20572410', '1920':'20572900', '1921':'20573386', '1922':'20573879', '1923':'20574338'}
	
    poem_lengths = []
    directory = 'c:\Users\Public\Documents\MyData\Poets\USData\PoetryTexts\\'
    for file in os.listdir(directory):
        file_num = re.sub(r'.txt', '', file)
        if int(file_num) <= int(poem_years[str(year)]) and int(file_num) > int(poem_years[str(year - 1)]):
            text = codecs.open(directory+file, "r", "utf-8")
            raw = text.read()
            raw = clean_poetry(raw, 0)
            length = len(raw)
            poem_lengths.append([file, length])  
    return poem_lengths   

#for other journals; just change directory to journal you are interested in
def get_poem_lengths(year):
    import os
    import pandas as pd
    poem_lengths = []
    journals = ["Century", "ContVerse", "Harpers", "LittleReview", "LyricWest", "Masses", "Midland", "Nation", "NewRepublic", "Others", "Scribners", "SmartSet"]
    for journal in journals:
        directory = 'c:\Users\Public\Documents\MyData\Poets\USData\JournalCorpora/' + journal + '\\'
        csv = pd.read_csv('c:\Users\Public\Documents\MyData\Poets\USData\JournalCorpora/' + journal + 'Data.csv')
        for file in os.listdir(directory):
            if csv[csv.FILENAME==file].YEAR.values == year: 
                text = codecs.open(directory+file, "r", "utf-8")
                raw = text.read()
                raw = clean_poetry(raw, 0)
                length = len(raw)
                poem_lengths.append([file, length])
    return poem_lengths   

def get_haiku_lengths():
    import os
    haiku_lengths = []
    directory = 'c:\Users\Public\Documents\MyData\HaikuArticle\Haikus\\'
    for file in os.listdir(directory):
        text = codecs.open(directory+file, "r", "utf-8")
        raw = text.read()
        raw = clean_poetry(raw)
        length = len(raw)
        haiku_lengths.append([file, length])  
    return haiku_lengths
		
#call this function when building corpus to reduce words in feature-set
def reduce_word_features(corpus, doc_terms, min_df):
    for text in corpus: 
        text[1][0] = {i:text[1][0][i] for i in text[1][0] if doc_terms[i] > min_df}
    return corpus
	
##############################################################################
# Functions to do normal 4-fold cross validation and randomized classification
##############################################################################
def cross_validate(iterations, haiku_labeled, short_labeled, corpus_length, journal):
    accuracy_scores = []
    most_informative = []
    mis_classified = []
    h_precision_scores = []
    h_recall_scores = []
    nh_precision_scores = []
    nh_recall_scores = []
    response = raw_input("Do you want to reduce dimensionality (y/n)?")
    if response == "y":
        min_df = raw_input("Include words occurring more than how many times?")
    #do normal cross-validation
    for i in range(iterations):
        haiku_random = []
        haiku_random = random.sample(haiku_labeled, corpus_length) #pick poems at random -- choose number based on smaller corpus size
        #Create 4 folds for validation testing
        cut_point = int((len(haiku_random))/4)
        hfold1 = haiku_random[0:cut_point]
        hfold2 = haiku_random[cut_point:(cut_point*2)]
        hfold3 = haiku_random[(cut_point*2):(cut_point*3)]
        hfold4 = haiku_random[(cut_point*3):]
        poetry_random = []
        poetry_random = random.sample(short_labeled, corpus_length) #draws this number of samples randomly from the feature-set; will have to adjust number according to corpus size
        cut_point2 = int((len(poetry_random))/4)
        pfold1 = poetry_random[0:cut_point2]
        pfold2 = poetry_random[cut_point2:(cut_point2*2)]
        pfold3 = poetry_random[(cut_point2*2):(cut_point2*3)]
        pfold4 = poetry_random[(cut_point2*3):]
        #build training and test-sets
        train_set = hfold1 + hfold2 + pfold1 + pfold2 + hfold3 + pfold3
        test_set =  hfold4 + pfold4 #+ hfold3 + pfold3
        #dimensionality reduction; shouldn't you be doing this for both train and test sets?
        if response =="y":
            doc_terms = documents_per_word(train_set + test_set)  #count how many documents each word appears in
            train_set = reduce_word_features((train_set + test_set), doc_terms, int(min_df))    #exclude terms that appear in min_df documents
        #train the classifier
        nb_classifier = NaiveBayesClassifier.train([e[1] for e in train_set])
        nb_classifier.labels()
        #check accuracy of classifier and store accuracy measure
        accuracy_scores.append(accuracy(nb_classifier, [el[1] for el in test_set]))
        #obtain the 30 most informative features for each iteration
        most_informative.append(nb_classifier.show_most_informative_features(n=30))
        #get haiku precision and recall measures from the test and store in list
        h_precision, h_recall, nh_precision, nh_recall = get_precision_recall(test_set, nb_classifier)
        h_precision_scores.append(h_precision)
        h_recall_scores.append (h_recall)
        nh_precision_scores.append(nh_precision)
        nh_recall_scores.append (nh_recall)
        #store list of mis-classified files from the journal corpus (i.e., files misclassified as haiku)
        for el in test_set:
            guess = nb_classifier.classify(el[1][0])
            if guess != el[1][1] and (re.findall(r'[a-z]', el[0][0]) or len(el[0]) > 8):  #this will exclude the haiku mis-classified as not-haiku
                mis_classified.append(el[0])
    #write the mis_classified texts to a file and print out the most-commonly mis-classified texts
    counter = collections.Counter(mis_classified)
    #prepare to print mis-classified files to .csv
    import csv
    filename = 'c:\Users\Public\Documents\MyData\HaikuArticle\errors.csv'
    mis_classified_texts = open(filename, 'a')
    wr = csv.writer(mis_classified_texts, quoting=csv.QUOTE_ALL)
    wr.writerow(counter.values())   #write the frequency
    wr.writerow(counter.keys())     #write the filenames
    #print the files to a document along with metadata
    print_misclassified_haiku(counter, journal)
    print("\nThe most commonly mis-classified files in this test were the following:")
    print counter.most_common()
    return accuracy_scores, most_informative, h_precision_scores, h_recall_scores, nh_precision_scores, nh_recall_scores 

def print_misclassified_haiku(counter, journal):
    import pandas as pd
    import operator
    sorted_counter = sorted(counter.iteritems(), key=operator.itemgetter(1))
    sorted_counter.reverse()
    if journal == "Poetry":
        csv = pd.read_csv('c:\Users\Public\Documents\MyData\Poets\USData\magazine.csv') #get Poetry metadata for printing purposes
        for item in sorted_counter: 
            fileid = item[0][:-4]
            last = csv[csv.id==int(fileid)].Last.values     #grab the last name of poet
            first = csv[csv.id==int(fileid)].First.values   #grab the first name
            year = csv[csv.id==int(fileid)].year.values     #grab the year
            directory = 'c:\Users\Public\Documents\MyData\Poets\USData\PoetryTexts\\' #point this to where your poetry texts are
            text = codecs.open(directory+fileid+'.txt', "r", "utf-8")
            raw = text.read()
            output_file = ('misclassified'+journal+'.txt')
            with open(output_file, mode = 'a') as f:
                f.write('\n' + item[0] + ': ' + str(last) + ',' + str(first) + '(' + str(year) + ')' + '\nMisclassified ' + str(item[1]) + ' times' + '\n' + raw + '\n')
    elif journal == "Combo":
        print "Can't write misclassified haiku to a file"    
    else:
        csv = pd.read_csv('c:\Users\Public\Documents\MyData\Poets\USData\JournalCorpora/' + journal + 'Data.csv')
        for item in sorted_counter:
            fileid = item[0]
            author = csv[csv.FILENAME==fileid].AUTHOR.values     #grab the last name of poet
            year = csv[csv.FILENAME==fileid].YEAR.values     #grab the year
            directory = 'c:\Users\Public\Documents\MyData\Poets\USData\JournalCorpora/' + journal + '\\' #point this to where your poetry texts are
            text = codecs.open(directory+fileid, "r", "utf-8")
            raw = text.read()
            output_file = ('misclassified'+journal+'.txt')
            with codecs.open(output_file, 'a', 'utf-8') as f:
                f.write('\n' + fileid + ': ' + str(author) + '(' + str(year) + ')' + '\nMisclassified ' + str(item[1]) + ' times' + '\n' + raw + '\n')

def get_precision_recall(test_set, classifier):
    import collections
    import nltk.metrics
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    i = 0
    for item in test_set:
        label = item[1][1]
        feature = item[1][0]
        refsets[label].add(i)                     #add the actual label of this instance
        observed = classifier.classify(feature)    #classify the feature-set
        testsets[observed].add(i)                  #record the result for this instance
        i += 1
    haiku_precision = nltk.metrics.precision(refsets['haiku'], testsets['haiku'])
    haiku_recall = nltk.metrics.recall(refsets['haiku'], testsets['haiku'])
    not_haiku_precision = nltk.metrics.precision(refsets['not-haiku'], testsets['not-haiku'])
    not_haiku_recall = nltk.metrics.recall(refsets['not-haiku'], testsets['not-haiku'])
    return haiku_precision, haiku_recall, not_haiku_precision, not_haiku_recall
    #print 'haiku F-measure:', nltk.metrics.f_measure(refsets['haiku'], testsets['haiku'])
    #print 'not-haiku precision:', nltk.metrics.precision(refsets['not-haiku'], testsets['not-haiku'])
    #print 'not-haiku recall:', nltk.metrics.recall(refsets['not-haiku'], testsets['not-haiku'])
    #print 'not-haiku F-measure:', nltk.metrics.f_measure(refsets['not-haiku'], testsets['not-haiku'])

def random_test(iterations, haiku_labeled, short_labeled, corpus_length):
    import random
    from nltk.classify import NaiveBayesClassifier  #import directly from .py file while in appropriate directory (then use updated function, which returns features as list)
    from nltk.classify.util import accuracy
    random_scores = []
    #need to mash the corpora together here, and then send the newly labeled corpora to the "for" loop
    #switch labels for half of each corpora
    haiku_a = haiku_labeled[0:int(len(haiku_labeled)/2)]
    haiku_b = haiku_labeled[int(len(haiku_labeled)/2):]
    for i in haiku_a:
        i[1][1] = 'not-haiku'
    poetry_a = short_labeled[0:int(len(short_labeled)/2)]
    poetry_b = short_labeled[int(len(short_labeled)/2):]
    for i in poetry_a:
        i[1][1] = 'haiku'
    #create new corpora based on these false labels
    haiku_sample = haiku_b + poetry_a    #i.e., real haiku and poetry labeled as haiku
    poetry_sample = haiku_a + poetry_b
    #now run the classification
    for i in range(iterations):
        #for i,k in zip(haiku_random[0::2], haiku_random[1::2]): #iterates through every other item
        #    i[1][1] = 'not-haiku'
        #for i,k in zip(poetry_random[0::2], poetry_random[1::2]):
        #    i[1][1] = 'haiku'
        haiku_random = []
        poetry_random = []
        haiku_random = random.sample(haiku_sample, corpus_length)
        poetry_random = random.sample(poetry_sample, corpus_length)
        cut_point = int(corpus_length/4)
        hfold1 = haiku_random[0:cut_point]
        hfold2 = haiku_random[cut_point:(cut_point*2)]
        hfold3 = haiku_random[(cut_point*2):(cut_point*3)]
        hfold4 = haiku_random[(cut_point*3):]
        pfold1 = poetry_random[0:cut_point]
        pfold2 = poetry_random[cut_point:(cut_point*2)]
        pfold3 = poetry_random[(cut_point*2):(cut_point*3)]
        pfold4 = poetry_random[(cut_point*3):]
        train_set = hfold1 + hfold2 + hfold3 + pfold1 + pfold2 + pfold3   
        test_set = hfold4 + pfold4 
        #train the classifier
        nb_classifier = NaiveBayesClassifier.train([e[1] for e in train_set])
        nb_classifier.labels()
        #check accuracy of classifier and store accuracy measure
        random_scores.append(accuracy(nb_classifier, [el[1] for el in test_set]))
    return random_scores

def get_top_features(most_informative):
    not_haiku = []
    haiku = []
    #get the top 10 predictors of non_haiku for each iteration of the classification tests
    for item in most_informative:
        i = 0
        for feature in item:
            if i <= 10:      #get no more than the top 10 words
                if feature[1] == 'not-haiku':
                    not_haiku.append(feature[0])
                    i += 1
    #get the top 10 predictors for haiku
    for item in most_informative:
        j = 0
        for feature in item:
            if j <= 10:    #get no more than the top 10 words
                if feature[1] == 'haiku':
                    haiku.append(feature[0])
                    j += 1
    fdist1 = FreqDist(not_haiku)  #get the frequency distribution of terms
    fdist2 = FreqDist(haiku)
    top_non_haiku = fdist1.keys()
    top_haiku = fdist2.keys()
    return top_non_haiku[:20], top_haiku[:20]
