#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:28:47 2017

@author: al
"""

import pandas as pd
import numpy as np
 
# get the csv
df = pd.read_csv("../csv_files/train_input_processed.csv")

project_2_dictionary = []

for index,row in df.iterrows():
    print index
    convo_list = row['conversation'].split(" ")
    for word in convo_list:
        if word not in project_2_dictionary:
            project_2_dictionary.append(word)
            
project_2_dictionary = {"word":np.array[sorted(project_2_dictionary)]}
                                        
out_df = pd.DataFrame(project_2_dictionary)

outfile = "../csv_files/dictionary.csv"

out_df.to_csv(outfile)

