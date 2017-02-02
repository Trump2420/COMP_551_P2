#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:28:47 2017

@author: al
"""

import pandas as pd
import numpy as np

# the minimum number of times a word must appear to be included
# in the dictionary
count_quota = 5
 
# get the csv
df = pd.read_csv("../csv_files/train_input_processed.csv")

project_2_dictionary = {}

# build the dictionary
for index,row in df.iterrows():
    print index
    # split up the conversation into strings
    convo_list = row['conversation'].split(" ")
    # add any unadded words
    for word in convo_list:
        if word not in project_2_dictionary.keys():
            project_2_dictionary[word] = 0
        else:
            project_2_dictionary[word] += 1
            
# create the dataframe that will hold the dictionary
minimum_dictionary = []
for word in project_2_dictionary:
    if project_2_dictionary[word] >= count_quota:
        minimum_dictionary.append(word)
                                          
out_df = pd.DataFrame({"word":np.array[sorted(minimum_dictionary)]})

outfile = "../csv_files/dictionary_minimum_count_{}.csv".format(count_quota)

out_df.to_csv(outfile)

