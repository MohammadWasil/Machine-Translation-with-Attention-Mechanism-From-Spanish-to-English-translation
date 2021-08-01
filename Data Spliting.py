# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 00:20:31 2021

@author: wasil
"""
import os
import pickle
from sklearn.model_selection import train_test_split

DIR_path = "D:\MSc Data Science\Advanced Modules\[INF-DSAM1B] Advanced Machine Learning B\Deep learning for NLP\Project\Machine translation with attention"
PROCESSED_DATA_PATH = "Processed Data"

with open(os.path.join(DIR_path, PROCESSED_DATA_PATH, "english_tokenized_tensor.pickle"), "rb") as f:
    english_tokenized_tensor = pickle.load(f)
    
with open(os.path.join(DIR_path, PROCESSED_DATA_PATH, "spanish_tokenized_tensor.pickle"), "rb") as f:
    spanish_tokenized_tensor = pickle.load(f)
    
# 70% training, 20% validation, 10% test data

en_train, en_valid, es_train, es_valid = train_test_split(english_tokenized_tensor, spanish_tokenized_tensor, test_size=0.2, random_state=8, 
                                         shuffle=True)

# len(en_train), len(en_valid), len(es_train), len(es_valid)
# (1363871, 340968, 1363871, 340968)

en_train, en_test, es_train, es_test = train_test_split(en_train, es_train, test_size=0.1, random_state=8, 
                                         shuffle=True)

# len(en_train), len(en_valid), len(en_test), len(es_train), len(es_valid), len(es_test)
# (1227483, 340968, 136388, 1227483, 340968, 136388)

with open('train.pickle', 'wb') as f:
    pickle.dump([en_train, es_train], f)
    
with open('valid.pickle', 'wb') as f:
    pickle.dump([en_valid, es_valid], f)
    
with open('test.pickle', 'wb') as f:
    pickle.dump([en_test, es_test], f)