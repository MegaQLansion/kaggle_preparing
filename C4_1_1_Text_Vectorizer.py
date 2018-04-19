# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:36:06 2018

@author: Lansion
"""
measurements=[{'city':'Dubai','temperature':33.},{'city':'London','temperature':12.},{'city':'San Fransisco','temperature':18.}]
#initialization
from sklearn.feature_extraction import DictVectorizer
VEC=DictVectorizer()

print VEC.fit_transform(measurements).toarray()
print VEC.get_feature_names()