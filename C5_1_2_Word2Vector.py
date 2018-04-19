# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:57:59 2018

@author: Lansion
"""

from sklearn.datasets import fetch_20newsgroups
news=fetch_20newsgroups(subset='all')
X,y=news.data,news.target

from bs4 import BeautifulSoup
import nltk,re
def news_to_sentences(news):
    news_text=BeautifulSoup(news).get_text()
    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences=tokenizer.tokenize(news_text)
    sentences=[]
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]','',sent.lower().strip()).split())
    return sentences

sentences=[]

for x in X:
    sentences+=news_to_sentences(x)
    
from gensim.models import word2vec

num_features=300
min_word_count=20
num_workers=2

context=5
downsampling=1e-3

model=word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)
model.init_sims(replace=True)
print 'The most close vocabulary to steve is',model.most_similar('steve')
