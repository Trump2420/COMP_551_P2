#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:50:51 2017

@author: al
"""

import pandas as pd
import numpy as np
from copy import copy

# get the conversations
df_train = pd.read_csv("../csv_files/train_input_processed.csv")

# the dataframe will be created from this
output = []

# get the dictionary
df_dictionary = pd.read_csv("../csv_files/dictionary.csv")

output = pd.DataFrame(pd.np.empty((len(df_train), len(df_dictionary))) * 0) 

## build up the template from the words in the project dictionary
#for index, row in df_dictionary.iterrows():
#    template[row['word']] = 0

# now go through each of the conversations
for index, row in df_train.iterrows():
    
#    # copy the template to reset each word's count at 0
#    conversation_count = copy(template)
    
    # get the list of words that appear in this conversation
    word_list = row['conversation'].split(" ")
    
    # go through each of the words and count how many times they show up
    for word in word_list:
        output.ix[index,df_dictionary[df_dictionary['word'] == word][0]] += 1
    
#    # add a copy of of the count to the output
#    output.append(copy(conversation_count))
    
# create a dataframe from the conversations counts
df = pd.DataFrame(output)

#save the dataframe in a csv
df.to_csv("../csv_files/word_counts.csv")