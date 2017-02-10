# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk import stem, pos_tag
import re
import csv


    
# words that we want to remove that are unlikely to show up in 
# stopwords but that we want to remove anyways
additional_words = ["</s>", "<number>", "</d>"]

# lemmatizer is needed to boil down words to their roots
wordnet_lemmatizer = stem.WordNetLemmatizer()

# a list of the common stopwords
sw = stopwords.words("english")



def filter_word(word):
    # don't keep strings with speaker
    if "<speaker_" in word:
        return ""
    
    # take out unnecesary additional words
    for ad_word in additional_words:
        if ad_word in word:
            #remove this substring
            word = word.replace(ad_word,"")
    
    # get rid of any leftover whitespace
    word = word.strip('\t\r\n')
    
    # remove all non-alphanumeric characters
    word = re.sub('[^0-9a-zA-Z]+', '', word)
    
    # don't add empty strings or stop words
    if word:
        return word
    else:
        return ""
        
def filter_conversation(conversation_list):
    conv_words_filtered = []

    for word in conversation_list:
        filtered_word = filter_word(word)
        if filtered_word:
            conv_words_filtered.append(filtered_word)
        
    return conv_words_filtered

    
def get_wordnet_pos(treebank_tag):
    # return the particle of speech which can be fed into the lemmatizer
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

        
def context_lemmatize(conversation_with_POS):
    # lemmatize the word given a specific part of speech
    # (i.e., verb, noun,adj,adv)
    
    lemmatized_conversation = []
    for word_with_POS in conversation_with_POS:
        POS = get_wordnet_pos(word_with_POS[1])
        if POS != '':
            lemmatized_conversation.append(
                        wordnet_lemmatizer.lemmatize(word_with_POS[0],POS))
        else:
            lemmatized_conversation.append(
                        wordnet_lemmatizer.lemmatize(word_with_POS[0]))
            
    return lemmatized_conversation
    
    
    
        
def get_ngrams(conversation_list,n_grams_length):
    # take the conversation and produce a list of the n-grams contained by
    # this conversation. For example
    # ["the", "quick", "brown"] returns 
    # ["the_quick", "quick_brown"] for n-gram of length 2
    
    # get a list of lists for the number of n_grams
    n_grams = zip(*[conversation_list[i:] for i in range(n_grams_length)])
        
    return ["_".join(n_gram) for n_gram in n_grams]
            
            
#class MySentences(object):
#    def __init__(self, fname):
#        self.fname = fname
# 
#    def __iter__(self):
#        for line in open("../csv_files/{}.csv".format(self.fname)):
#            # return a generator, removing the id and blank space that
#            # arrives from splitting
#            if (re.split("\W+",line.lower())[0] != 'id') and line:
#                yield line.lower().split(" ")[1:]
    
    

if __name__ == "__main__":
    
    n_grams_count = 1 # if 1, just the words themselves
    
    for test_train in ["test","train"]:
    
        # get the training data
        conv_df = pd.read_csv('../csv_files/{}_input.csv'.format(test_train))
        
        fname = "../csv_files/{0}_input_processed_{1}_grams.csv".format(
                  test_train,n_grams_count)
        
        with open(fname,'w') as csvfile:
            
            writer = csv.writer(csvfile)
            writer.writerow(('id','conversation'))
            
            #iterate through the list of posts
            for index,row in conv_df.iterrows():
                print index
                
                # remove the noisy words/characters
                conversation_list = filter_conversation(
                                        row['conversation'].split(" "))  
                
                # find the part of speach for each word in the conversation
                conversation_list_wth_POS = pos_tag(conversation_list)
                
                # lemmatize the conversation
                conversation_list = context_lemmatize(conversation_list_wth_POS)
                
                # now remove stopwords
                conversation_list = [x for x in conversation_list if x not in sw]
    
                if n_grams_count > 1:
                    conversation_list.extend(get_ngrams(conversation_list,n_grams_count)) 
                    
                writer.writerow((index," ".join(conversation_list))) 