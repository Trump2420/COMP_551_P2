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
     
    
class TfidfW2V():
    def __init__(self,trigram,model=None,tfidf=None):
        # load in the trigram phraser and any existing models
        self.tg = trigram
        if tfidf:
            self.idf = tfidf
            max_idf = max(self.idf.idf_)
        
            self.word_to_idf = defaultdict(
                lambda: max_idf,
                [(w, self.idf.idf_[i]) for w, i in self.idf.vocabulary_.items()])
        else:
            self.idf = None
        if model:
            self.w2v_model = model
        else:
            self.w2v_model = None
        self.word_to_idf = None
        self.dim = 200
        
        
    def fit(self,corpus):
        # train whichever model is lacking
        if (not self.w2v_model) or (not self.idf):
            print "getting test corpus"
            tg_corpus_test = [[x for x in list(self.tg[conv]) if x not in sw] for conv in corpus["test"]]
            print "getting training corpus"
            tg_corpus_train = [[x for x in list(self.tg[conv]) if x not in sw] for conv in corpus["train"]]
            
            if (not self.w2v_model):       
                """
                this is an incredibly important step in word2vec. you can't just
                build a vocabulary and stop there. The more you train the 
                model on corpi, the better the model gets. This can take quite
                a while however. W2V expects lists of lists of words as input
                """
                self.w2v_model = gs.models.Word2Vec(iter=5,size=200,workers=4,
                                                    min_count=10)
                self.w2v_model.build_vocab(tg_corpus_train)
                
                for i in range(3):
                    print i, " test"
                    self.w2v_model.train(tg_corpus_test)
                    print i, " train"
                    self.w2v_model.train(tg_corpus_train)
                
            if not self.idf:  
                # in contrast to w2v, tfidf expects list of strings and only
                # needs to be fit once
                tfidf_corpus = [" ".join(conv) for conv in tg_corpus_test + 
                                tg_corpus_train]
                     
                
                self.idf = TfidfVectorizer(analyzer='word').fit(tfidf_corpus)
                
                max_idf = max(self.idf.idf_)
                
                # the dictionary that will return the idf value for a word.
                # if a word is not in the dictionary for tfidf, the maximum
                # idf value is returned since this word is at least as rare
                # as the rarest word in the corpus
                self.word_to_idf = defaultdict(
                    lambda: max_idf,
                    [(w, self.idf.idf_[i]) for w, i in 
                     self.idf.vocabulary_.items()])
        
        
    def vectorize_conversation(self,conv):
        # take a list of words in a conversation and return an averaged value
        # based on word2vec vector representation of this word and the weight
        # attributed to it by tfidf
        
        #get a dictionary of the count of each unique word in the conversation
        counts = Counter(conv)
        # the total number of words in the conversation
        total = float(len(conv))
        
        # if a word is not in the w2v vocabulary, return a 0 vector, otherwise
        # get the tf, idf and w2v vector representation and multiply all of
        # these together. 
        vector = np.mean(
                    [(counts[w]/total)*self.word_to_idf[w]*self.w2v_model[w] for w in counts
                     if w in self.w2v_model] or [np.zeros(self.dim)], axis = 0)
        return vector
        
    

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
            
            #don't take stopwords just yet, because they may be something like
            # hall_<of>_fame which is important
            
            #add the list in one string to the corpus
            corpus[test_train].append(conversation_list)
            
    # learn the bigrams and trigrams from the text
    bigram = gs.models.Phrases(corpus["test"] + corpus["train"])
    trigram = gs.models.Phrases(bigram[corpus["test"] + corpus["train"]]) 

    
    # create the vectorizer
    vectorizer = TfidfW2V(trigram)
    # feed the corpus of train and test sets into the vectorizer
    vectorizer.fit(corpus)
    
    """
    note that the above can be accomplished with pre-existing tfidf and word2vec
    objects like
    
    vectorizer = TfidfW2V(trigram,w2vmodel,tfidf)
    
    so these can be saved and loaded in other scripts to save time
    """
    
    # save the results for each files
    for item in  ["train","test"]:
        
        fname = "../csv_files/{}_input_vectorized.csv".format(item)
        
        with open(fname,'w') as csvfile:
            headers= ['id']
            headers.extend(range(200))
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for i in range(len(corpus[item])):
                print i
                next_row = [i]
                vector = vectorizer.vectorize_conversation(corpus[item][i])
                next_row.extend(["{0:.4f}".format(x) for x in list(vector)])
                writer.writerow(next_row) 
        
    

                