# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:29:07 2021

@author: wasil
"""
import os
import re
import torch
import nltk
from tqdm import tqdm
import pickle


DIR_path = "D:\MSc Data Science\Advanced Modules\[INF-DSAM1B] Advanced Machine Learning B\Deep learning for NLP\Project\Machine translation with attention"
english_data_path = "Data\\es-en"
spanish_data_path = "Data\\es-en"

print("reading the data...")
# READ THE DATA
# 'utf-8' removes b'' character string literal
# splitlines() remove newline character
with open(os.path.join(DIR_path, english_data_path, "europarl-v7.es-en.en"), "rb") as f:
    content_english = f.read().decode("utf-8").splitlines()

with open(os.path.join(DIR_path, spanish_data_path, "europarl-v7.es-en.es"), "rb") as f:
    content_spanish = f.read().decode("utf-8").splitlines()

_patterns = [r'\'',
             r'\"',
             r'\.',
             r'<br \/>',
             r',',
             r'\(',
             r'\)',
             r'\!',
             r'\?',
             r'\;',
             r'\:',
             r'\s+']

_replacements = [' \'  ',
                 '',
                 ' . ',
                 ' ',
                 ' , ',
                 ' ( ',
                 ' ) ',
                 ' ! ',
                 ' ? ',
                 ' ',
                 ' ',
                 ' ']

_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

def sentence_preprocess(sentence):
    """https://pytorch.org/text/_modules/torchtext/data/utils.html"""
    sentence = sentence.lower()
    for pattern_re, replaced_str in _patterns_dict:
        sentence = pattern_re.sub(replaced_str, sentence)
    
    return sentence

print("Processing the English data...")
# preprocess the english sentence
sentence_english = []
for sent in content_english:
    sentence_english.append(sentence_preprocess(sent))
print("total english sentences: ", len(sentence_english))

print("Tokenizing the English dataset ...")
# Loop over wach of the sentence and tokenize eacch sentenec separately.
# will take some time to tokenize each sentence.
english_tokenized_text = [ nltk.word_tokenize(sentence_english[i], language="english") for i in tqdm(range(len(sentence_english))) ]

# create word index
# assign each word a number.
word_to_index = {}
words=[]
for sentence in english_tokenized_text:
    for word in sentence:
        words.append(word)
UNIQUE_WORDS = set(words)

for index, word in enumerate(UNIQUE_WORDS):
    word_to_index[word] = index

# add tokens: <SOS> and <EOS>
word_to_index["<SOS>"] = list(word_to_index.values())[-1] + 1
word_to_index["<EOS>"] = list(word_to_index.values())[-1] + 1

# using word index, create tensor
# convert each of the sentence into numbers.
english_tokenized_tensor = []

for sentence in english_tokenized_text:
    #english_tokenized_tensor.append( [word_to_index[word] for word in sentence]  )
    tensor_list=[]
    tensor_list.append(word_to_index["<SOS>"])
    tensor_list = tensor_list + [word_to_index[word] for word in sentence]
    tensor_list.append(word_to_index["<EOS>"])

    english_tokenized_tensor.append(torch.tensor(tensor_list, dtype=torch.long))

# shorten the sentence with words > 50 - ENGLISH
MAX_SENTENCE_LENGTH = 50
for i in range(len((english_tokenized_tensor))):
    if(len(english_tokenized_tensor[i]) > MAX_SENTENCE_LENGTH):
        #print(data_english_tensor[i])
        english_tokenized_tensor[i] = english_tokenized_tensor[i][0:MAX_SENTENCE_LENGTH]
# pad each of the tensors.
english_tokenized_tensor = torch.nn.utils.rnn.pad_sequence(english_tokenized_tensor, batch_first =True)

print("Saving ...")
# save the data in pickle file
with open('data_english_FULL.pickle', 'wb') as data_file:
    pickle.dump(english_tokenized_tensor, data_file, protocol=pickle.HIGHEST_PROTOCOL) 
    
print("Processing the Spanish data...")
# preprocess the spanish sentence
sentence_spanish = []
for sent in content_spanish:
    sentence_spanish.append(sentence_preprocess(sent))
print("total spanish sentences: ", len(sentence_spanish))
    
print("Tokenizing the Spanish dataset ...")
# Loop over wach of the sentence and tokenize eacch sentenec separately.
# will take some time to tokenize each sentence.
spanish_tokenized_text = [ nltk.word_tokenize(sentence_spanish[i], language="spanish") for i in tqdm(range(len(sentence_spanish))) ]

# create word index
# assign each word a number.
word_to_index = {}
words=[]
for sentence in spanish_tokenized_text:
    for word in sentence:
        words.append(word)
UNIQUE_WORDS = set(words)

for index, word in enumerate(UNIQUE_WORDS):
    word_to_index[word] = index

# add tokens: <SOS> and <EOS>
word_to_index["<SOS>"] = list(word_to_index.values())[-1] + 1
word_to_index["<EOS>"] = list(word_to_index.values())[-1] + 1

# using word index, create tensor
# convert each of the sentence into numbers.
spanish_tokenized_tensor = []

for sentence in spanish_tokenized_text:
    tensor_list=[]
    tensor_list.append(word_to_index["<SOS>"])
    tensor_list = tensor_list + [word_to_index[word] for word in sentence]
    tensor_list.append(word_to_index["<EOS>"])
    spanish_tokenized_tensor.append(torch.tensor(tensor_list, dtype=torch.long))

# shorten the sentence with words > 50 - SPANISH
for i in range(len((spanish_tokenized_tensor))):
    if(len(spanish_tokenized_tensor[i]) > MAX_SENTENCE_LENGTH):
        #print(data_english_tensor[i])
        spanish_tokenized_tensor[i] = spanish_tokenized_tensor[i][0:MAX_SENTENCE_LENGTH]
# pad each of the tensors.
spanish_tokenized_tensor = torch.nn.utils.rnn.pad_sequence(spanish_tokenized_tensor, batch_first =True)

print("Saving ...")
# save the data in pickle file    
with open('data_spanish_FULL.pickle', 'wb') as data_file:
    pickle.dump(spanish_tokenized_tensor, data_file, protocol=pickle.HIGHEST_PROTOCOL)