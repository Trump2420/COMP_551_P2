# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk import stem, pos_tag
import re


    
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
    # lemmatize the word given a specific context (i.e., verb, noun,adj,adv)
    
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
        

if __name__ == "__main__":

    test_train = "train"
    # get the training data
    df = pd.read_csv('../csv_files/{}_input.csv'.format(test_train))  
    
    #iterate through the list of posts
    for index, row in df.iterrows():
            
        print index
        
        # get each of the words in the conversation
        conversation_list = row['conversation'].split(" ")
        
        # remove the noisy words/characters
        conversation_list = filter_conversation(conversation_list)  
        
        # find the part of speach for each word in the conversation
        conversation_list_wth_POS = pos_tag(conversation_list)
        
        # lemmatize the conversation
        conversation_list = context_lemmatize(conversation_list_wth_POS)
        
        # now remove stopwords
        conversation_list = [x for x in conversation_list if x not in sw]
            
        df.set_value(index,'conversation'," ".join(conversation_list))
        
    df.to_csv("../csv_files/{}_input_processed.csv".format(test_train), index=False)        