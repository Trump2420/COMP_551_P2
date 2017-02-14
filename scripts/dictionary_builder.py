#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:28:47 2017

@author: al
"""

import pandas as pd
#import numpy as np
import nltk
import re

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open("../csv_files/{}.csv".format(self.fname)):
            # return a generator, removing the id and blank space that
            # arrives from splitting
            if re.split("\W+",line.lower())[0] != 'id':
                yield re.split("\W+",line.lower())[1:-1]

def build_dictionary():
    # get the csv containing all the preprocessed conversations in the training data
    train_csv = "train_input_processed_trigrams"
    test_csv = "test_input_processed_trigrams"
    
    output_csv = "../csv_files/dictionary_frequency_trigrams.csv"
    
    
    
    tokens = []
    for fname in [train_csv,test_csv]:
        #get all the words that appear in the training corpus
        i = 0
        for conversation in MySentences(fname):
            print i
            i+=1
            tokens.extend(conversation)
    
    
    # find the frequency distribution of these tokens
    freq_dist = nltk.FreqDist(tokens)
    
    # sort by order of frequency
    most_common = freq_dist.most_common()
    
    # obtains the frequency and words
    words = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]
    
    # create a dataframe and save as a csv                                       
    out_df = pd.DataFrame({"word":words, "counts":counts})
    out_df.to_csv(output_csv, index=False)
    
if __name__ == "__main__":
    build_dictionary()

