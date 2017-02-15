#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:50:51 2017

@author: al
"""

#

import pandas as pd
import numpy as np
from copy import copy
import dictionary_builder
import preprocessing_n_grams
from collections import OrderedDict
import csv

#preprocessing_n_grams.get_n_grams(3)
#
#dictionary_builder.build_dictionary()

# the top n most popular words to use
most_common_count = 2000

# get the dictionary with only the desired number of most common words
df_dictionary = pd.read_csv("../csv_files/dictionary_frequency_trigrams.csv").head(most_common_count)
template = OrderedDict()
## build up the template from the words in the project dictionary
for index, row in df_dictionary.iterrows():
    template[row['word']] = 0


headers= ["id"]
headers.extend(template.keys())



# vectorize either the train or test sets
for test_train in ["test","train"]:
    outfile = "../csv_files/{0}_trigram_vectors_{1}.csv".format(test_train,most_common_count)
    
    # get the conversations
    df_in = pd.read_csv("../csv_files/{}_input_processed_trigrams.csv".format(test_train))
    
    

    
    # the dataframe will be created from this
    
    with open(outfile, 'w') as csvfile:
                
        writer = csv.writer(csvfile)
        writer.writerow(headers)
    
    # make the table that will hold the vectors
    #output = pd.DataFrame(pd.np.empty((len(df_train), most_common_count)) * 0) 
#        output = []
        # now go through each of the conversations
        for index, row in df_in.iterrows():
            print index
            counts = copy(template)
            output = [str(index)]
        #    # copy the template to reset each word's count at 0
            
            # get the list of words that appear in this conversation
            word_list = row['conversation'].split(" ")
            
            # go through each of the words and count how many times they show up
            for word in word_list:
                if any(df_dictionary.word == word):
                    counts[word] += 1

            
            # add a copy of of the count to the output
            output.extend(counts.values())
            
            # save the vector, reducing memory usage
            if index%10000 == 0:   
                print test_train + ": " + str(index)
            
            writer.writerow(output)
        
## create a dataframe from the conversations counts
#df = pd.DataFrame(output)
#
##save the dataframe in a csv
#df.to_csv("../csv_files/word_counts.csv")