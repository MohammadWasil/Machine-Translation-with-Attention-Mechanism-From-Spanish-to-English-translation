import numpy as np
import os
import nltk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
nltk.download('punkt')
from utils_io import read_data

import spacy

import pandas as pd

spacy_es = spacy.load('es_core_news_sm')
#spacy_es = spacy.load('es')
spacy_en = spacy.load('en_core_web_sm')
#spacy_en = spacy.load('en')

def sentence_preprocess(content_english, content_spanish):
    print("preprocessing the sentences ...")

    # first, we will remove all the sentence having length greater than 50.
    # removing sentences having length > 50
    english_tokenized_text = []
    spanish_tokenized_text = []
    
    for i in tqdm(range(len(content_english))):
        tok_eng = nltk.word_tokenize(content_english[i], language="english")
        tok_esp = nltk.word_tokenize(content_spanish[i], language="spanish") 
    
        # both of the sentence in english and spanish should be smaller than 48
        if (len(tok_eng) <= 50) and (len(tok_esp) <= 50):
            english_tokenized_text.append(tok_eng)
            spanish_tokenized_text.append(tok_esp)
            #tqdm._instances.clear()
            
    # sort the data, so we have maximum words in a sentence in the top most sentences, and not padded sentence most of the time.
    # sentence with small number of words would be at the bottom, which we will discard, eventually, since we will select top
    # 1,000,000 sentences for training.
    lenlist=[]   
    for x in english_tokenized_text:
        lenlist.append(len(x))   
    
    sortedindex = np.argsort(lenlist)[::-1]
    lst_eng = ['english']*len(english_tokenized_text)
    lst_spn = ['spanish']*len(spanish_tokenized_text)
    
    for i in range(len(english_tokenized_text)):    
        # placing element in the lst2 list by taking the
        # value from original list lst where it should belong 
        # in the sorted list by taking its index from sortedindex
        lst_eng[i] = english_tokenized_text[sortedindex[i]] 
        lst_spn[i] = spanish_tokenized_text[sortedindex[i]] 
    
    # considering only first 1_000_000 sentences.
    english_tokenized_text = lst_eng[0:1_000_000]
    spanish_tokenized_text = lst_spn[0:1_000_000]
    print("Processing completed")
    print("Splitting the data ...")
    en_train, en_valid, es_train, es_valid = train_test_split(english_tokenized_text, spanish_tokenized_text, test_size=0.1, random_state=False, 
                                             shuffle=False)
    
    
    en_valid, en_test, es_valid, es_test = train_test_split(en_valid, es_valid, test_size=0.2, random_state=False, 
                                             shuffle=False)
    
    # len(en_valid), len(es_valid), len(en_test), len(es_test), len(en_train), len(es_train)
    # (80000, 80000, 20000, 20000, 900000, 900000)
    
    raw_data = {'src': [' '.join( line) for line in es_train], 'trg': [' '.join( line) for line in en_train]}
    train_data = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    raw_data = {'src': [' '.join( line) for line in es_valid], 'trg': [' '.join( line) for line in en_valid]}
    valid_data = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    raw_data = {'src': [' '.join( line) for line in es_test], 'trg': [' '.join( line) for line in en_test]}
    test_data = pd.DataFrame(raw_data, columns=["src", "trg"])


    # save the data. The data will be saved only once.    
    train_data.to_csv("train.csv", index=False)
    valid_data.to_csv("val.csv", index=False)
    test_data.to_csv("test.csv", index=False)

def tokenize_es(text):
    return [tok.text for tok in spacy_es.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def prepare_data():
    # if the process file already exists, then there is no need to process and split the data again.
    if((os.path.isfile('train.csv') == False) or (os.path.isfile('val.csv') == False) or (os.path.isfile('test.csv') == False)):
        
        content_english, content_spanish = read_data()
        sentence_preprocess(content_english, content_spanish)
