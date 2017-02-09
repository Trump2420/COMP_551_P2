#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:28:47 2017

@author: al
"""

import pandas as pd
#import numpy as np
import nltk


# get the csv containing all the preprocessed conversations in the training data
input_csv = "../csv_files/train_input_processed.csv"

output_csv = "../csv_files/train_dictionary.csv"

 
# get the csv
df = pd.read_csv(input_csv)

tokens = []

#get all the words that appear in the training corpus
for index,row in df.iterrows():
    print index
    tokens.extend(row['conversation'].split(" "))


project_2_dictionary = []


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

