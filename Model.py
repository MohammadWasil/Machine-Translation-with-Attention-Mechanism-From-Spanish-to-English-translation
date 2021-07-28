# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 00:56:51 2021

@author: wasil
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk
from tqdm import tqdm
import pickle

# For cuda.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 50
EMBEDDING_DIM = 30
EPOCHS = 1
EMBEDDING_SIZE_ENGLISH = len(word_to_index_eng)
EMBEDDING_SIZE_SPANISH = len(word_to_index_spanish)

class Encoder(nn.Module):
    def __init__(self, EMBEDDING_DIM, EMBEDDING_SIZE_ENGLISH):
        super(Encoder, self).__init__()
        self.embedding_dim = EMBEDDING_DIM
        self.embedding_size_english = EMBEDDING_SIZE_ENGLISH
        
        self.embedding = nn.Embedding( self.embedding_size_english , self.embedding_dim)
        
    
    def forward(self, inputs):
        # WRITE CODE HERE
        # input : [sequence_len, batch_size]
        embeds = self.embedding(inputs)
        # https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
        # output of the embedding is [seq_len, batch_size, embedding_size]
        
        
        return embeds

def padding(english_tokenized_tensor, spanish_tokenized_tensor):
    
    spanish_batch=[]
    for batch in spanish_tokenized_tensor:
        if(len(batch) > MAX_SENTENCE_LENGTH):
            batch = batch[0:MAX_SENTENCE_LENGTH]
        if( (MAX_SENTENCE_LENGTH - batch.shape[0]) != 0 ) :
            spanish_batch.append(torch.cat((batch, torch.LongTensor([0]).repeat(MAX_SENTENCE_LENGTH - len(batch))), dim=0) )
        else:
            spanish_batch.append(batch)
       
    english_batch=[]
    for batch in english_tokenized_tensor:
        # only consider top 50 words in each sentences.
        if(len(batch) > MAX_SENTENCE_LENGTH):
            batch = batch[0:MAX_SENTENCE_LENGTH]
        if( (MAX_SENTENCE_LENGTH - batch.shape[0]) != 0 ) :
            english_batch.append(torch.cat((batch, torch.LongTensor([0]).repeat(MAX_SENTENCE_LENGTH - len(batch))), dim=0) )
        else:
            english_batch.append(batch)

    # the first element of the batchholds the first elements of every sequence in the batch. 
    # the second element of the batch is going to hold the second element of every sequence in the batch, and so on.
    # shape : [sequence_length, batch_size]
    spanish_batch = torch.transpose(torch.stack(spanish_batch), 0, 1)
    english_batch = torch.transpose(torch.stack(english_batch), 0, 1)
    
    return english_batch, spanish_batch

if __name__=='__main__':
    # encoder model
    model = Encoder(EMBEDDING_DIM, EMBEDDING_SIZE_ENGLISH)
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    loss_function = nn.NLLLoss()

    for epochs in range(EPOCHS):
        total_loss = 0
        model.to(device)
        for i in range(0, len(english_tokenized_tensor), BATCH_SIZE):
            english_batch = english_tokenized_tensor[i:i+BATCH_SIZE] 
            spanish_batch = spanish_tokenized_tensor[i:i+BATCH_SIZE] 
    
            en, es = padding(english_batch, spanish_batch) # shape : [sequence_length, batch_size]
    
            en = en.to(device)
            es = es.to(device)
    
            en_prediction = model(en)
            




