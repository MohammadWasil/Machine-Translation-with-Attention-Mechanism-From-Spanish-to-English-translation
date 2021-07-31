# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 23:38:40 2021

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

print("Reading the file...")
with open(os.path.join(DIR_path, spanish_data_path, "europarl-v7.es-en.es"), "rb") as f:
    content_spanish = f.read().decode("utf-8").splitlines()
    
def sent_preprocess(sentence):
    sentence=sentence.lower()             
    sentence = re.sub(r"[-,.!?()]+", r"", sentence)
    return sentence

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

print("Processing the file...")
# preprocess the spanish sentence
sentence_spanish = []
for sent in content_spanish:
    sentence_spanish.append(sentence_preprocess(sent))
print("total spanish sentences: ", len(sentence_spanish))

print("Tokenizing the file...")
# Loop over wach of the sentence and tokenize eacch sentenec separately.
# will take some time to tokenize each sentence.
spanish_tokenized_text = [ nltk.word_tokenize(sentence_spanish[i], language="spanish") for i in tqdm(range(len(sentence_spanish))) ]

# create word index
# assign each word a number.
word_to_index_spanish = {}
words=[]
for sentence in spanish_tokenized_text:
    for word in sentence:
        words.append(word)
UNIQUE_WORDS = set(words)

for index, word in enumerate(UNIQUE_WORDS):
    word_to_index_spanish[word] = index

# add tokens: <SOS> and <EOS>
word_to_index_spanish["<SOS>"] = list(word_to_index_spanish.values())[-1] + 1
word_to_index_spanish["<EOS>"] = list(word_to_index_spanish.values())[-1] + 1

# using word index, create tensor
# convert each of the sentence into numbers.
spanish_tokenized_tensor = []

for sentence in spanish_tokenized_text:
    
    tensor_list=[]
    tensor_list.append(word_to_index_spanish["<SOS>"])
    tensor_list = tensor_list + [word_to_index_spanish[word] for word in sentence]
    tensor_list.append(word_to_index_spanish["<EOS>"])
    spanish_tokenized_tensor.append(torch.tensor(tensor_list, dtype=torch.long))

print("Saving the file...")
# spanish_tokenized_tensor
import pickle
with open('spanish_tokenized_tensor.pickle', 'wb') as handle:
    pickle.dump(spanish_tokenized_tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('word_to_index_spanish.pickle', 'wb') as handle:
    pickle.dump(word_to_index_spanish, handle, protocol=pickle.HIGHEST_PROTOCOL)
