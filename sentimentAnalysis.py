# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:13:56 2020

@author: Serdar
"""


import pandas as pd       
data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
 


 
import random
sentiment_data = list(zip(data["review"], data["sentiment"]))
random.shuffle(sentiment_data)

train_X, train_y = list(zip(*sentiment_data[:20000]))
 

test_X, test_y = list(zip(*sentiment_data[20000:]))

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
 
 
lemmatizer = WordNetLemmatizer()
 
 
def penn_to_wn(tag):
   
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
 
 
def clean_text(text):
    text = text.replace("<br />", " ")
    return text
 
 
def swn_polarity(text):
 
 
    sentiment = 0.0
    tokens_count = 0
 
    text = clean_text(text)
 
 
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
 
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
 
            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
 
         
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
 
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1

    if not tokens_count:
        return 0

    if sentiment >= 0:
        return 1
 

    return 0


from sklearn.metrics import accuracy_score
pred_y = [swn_polarity(text) for text in test_X]
 
print ("AUC: ",accuracy_score(test_y, pred_y)) 