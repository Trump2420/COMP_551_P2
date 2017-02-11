# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk import stem, pos_tag
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim as gs
from collections import defaultdict,Counter
import numpy as np
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
                        str(wordnet_lemmatizer.lemmatize(word_with_POS[0],POS)))
        else:
            lemmatized_conversation.append(
                        str(wordnet_lemmatizer.lemmatize(word_with_POS[0])))
            
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
#    def __init__(self):
#        pass
# 
#    def __iter__(self):
#        for line in open("../csv_files/{}.csv".format(self.fname)):
#            # return a generator, removing the id and blank space that
#            # arrives from splitting
#            if (re.split("\W+",line.lower())[0] != 'id') and line:
#                yield line.lower().split(" ")[1:-1]
#
#class list_to_string(conv_list):
#    for 
    
class TfidfW2V():
    def __init__(self,trigram):
        self.tg = trigram
        self.idf = None
        self.w2v_model = None
        self.word_to_idf = None
        self.dim = 200
        
        
    def fit(self,corpus):
        print "getting test corpus"
        tg_corpus_test = [[x for x in list(self.tg[conv]) if x not in sw] for conv in corpus["test"]]
        print "getting training corpus"
        tg_corpus_train = [[x for x in list(self.tg[conv]) if x not in sw] for conv in corpus["train"]]
        
        self.w2v_model = gs.models.Word2Vec(iter=5,size=200,workers=4,min_count=10)
        self.w2v_model.build_vocab(tg_corpus_train)
        
        for i in range(3):
            print i, " test"
            self.w2v_model.train(tg_corpus_test)
            print i, " train"
            self.w2v_model.train(tg_corpus_train)
            
                        
        tfidf_corpus = [" ".join(conv) for conv in tg_corpus_test + tg_corpus_train]
                        
#        self.w2v_model = gs.models.Word2Vec(iter=5,size=self.dim,workers=4,min_count = 5)
#        self.w2v_model.build_vocab(tg_corpus)
        
        self.idf = TfidfVectorizer(analyzer='word').fit(tfidf_corpus)
        
        max_idf = max(self.tf.idf_)
        
        self.word_to_idf = defaultdict(
            lambda: max_idf,
            [(w, self.idf.idf_[i]) for w, i in self.idf.vocabulary_.items()])
        
        
    def vectorize_conversation(self,conv):
        #get a dictionary of the count of each unique word in the conversation
        counts = Counter(conv)
        # the total number of words in the conversation
        total = float(len(conv))
        vector = np.mean(
                    [(counts[w]/total)*self.word_to_idf[w]*self.w2v_model[w] for w in counts
                     if w in self.w2v_model] or [np.zeros(self.dim)], axis = 0)
        return vector
        
class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
 
    def __iter__(self):
        for line in open("{}.csv".format(self.fname)):
            # return a generator, removing the id and blank space that
            # arrives from splitting
            if re.split("\W+",line.lower())[0] != 'id':
                yield re.split("\W+",line.lower())[1:-1]
    

if __name__ == "__main__":
    
    corpus = {"train":[],"test":[]}

    n_grams_count = 2
    
    
    for test_train in ["train","test"]:
    
        conversations_df = pd.read_csv('../csv_files/{}_input.csv'.format(test_train))
        count = 0
        for index,row in conversations_df.iterrows():
            print test_train,": ", index
            # remove the noisy words/characters
            conversation_list = filter_conversation(row['conversation'].split(" "))  
            
            # find the part of speach for each word in the conversation
            conversation_list_wth_POS = pos_tag(conversation_list)
            
            # lemmatize the conversation
            conversation_list = context_lemmatize(conversation_list_wth_POS)  
            
#            conversation_list = [x for x in conversation_list if x not in sw]
            #don't take stopwords out now, because they may be something like
            # hall_<of>_fame which is important
            #add the list in one string to the corpus
            corpus[test_train].append(conversation_list)
            
    # learn the bigrams and trigrams from the text
    bigram = gs.models.Phrases(corpus["test"] + corpus["train"])
    trigram = gs.models.Phrases(bigram[corpus["test"] + corpus["train"]]) 

#    df = pd.DataFrame({"id":range(5001),"conversation": trigram[corpus['train']]})
#    df.to_csv('train.csv')
#    
#    df = pd.DataFrame({"id":range(5001),"conversation": trigram[corpus['test']]})
#    df.to_csv('test.csv')
#    model = gs.models.Word2Vec(iter=5,size=200,workers=4,min_count=5)
#    model.build_vocab(corpus["train"])
#    for i in range(3):
##        sentences = MySentences('test_input_processed_1_grams')
#        model.train(list(trigram[corpus['test']]))
##        sentences = MySentences('train_input_processed_1_grams')
##        model.train(trigram[corpus['train']])
    vectorizer = TfidfW2V(trigram)
#            
#    # feed the conversations with trigrams into the tfidf    
#   
    vectorizer.fit(corpus)
    
    

                