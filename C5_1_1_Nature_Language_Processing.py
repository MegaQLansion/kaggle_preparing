# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:43:35 2018

@author: Lansion
"""
sent1='Sunziyang is a smelly pig'
sent2='Who can not find a job'
#split sentences
import nltk
tokens_1=nltk.word_tokenize(sent1)
print 'The split token is',tokens_1
#sequence as ASCII
vocab_1=sorted(set(tokens_1))
print 'The sequenced token is',vocab_1
#looking for the original vocabulary root
stemmer=nltk.stem.PorterStemmer()
stem_1=[stemmer.stem(t) for t in tokens_1]
print 'The stemmed vocabulary is',stem_1

#part in sentence
pos_tag_1=nltk.tag.pos_tag(tokens_1)
print 'The part of vocabulary in sentence is',pos_tag_1