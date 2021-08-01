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
import os


class Encoder(nn.Module):
    def __init__(self, EMBEDDING_DIM, EMBEDDING_SIZE_ENGLISH, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM):
        super(Encoder, self).__init__()

        self.embedding_dim = EMBEDDING_DIM
        self.embedding_size_english = EMBEDDING_SIZE_ENGLISH
        self.hidden_dim = ENCODER_HIDDEN_DIM

        self.embedding = nn.Embedding( self.embedding_size_english , self.embedding_dim)

        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, bidirectional =True)

        self.linear = nn.Linear(self.hidden_dim*2, DECODER_HIDDEN_DIM)

    def forward(self, inputs):
        # WRITE CODE HERE
        # input : [sequence_len, batch_size]
        embeds = self.embedding(inputs)
        # https://stackoverflow.com/questions/49466894/how-to-correctly-give-inputs-to-embedding-lstm-and-linear-layers-in-pytorch
        # output of the embedding is [seq_len, batch_size, embedding_size]

        #input: [seq_len, batch_size, embedding_size]
        rnn_enc, hidden = self.rnn(embeds)
        # rnn out shape : [sequence_len, batch_size, ENCODER_HIDDEN_DIM*2]
        # hidden shape   : [2, batch_size, ENCODER_HIDDEN_DIM]

        # concatenate both forward and backward hidden vectors
        hidden_f_b = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim = 1)
        # output shape: [batch_size, ENCODER_HIDDEN_DIM*2]

        # input: [batch_size, ENCODER_HIDDEN_DIM*2]
        hidden_enc = self.linear(hidden_f_b)
        # output: [batch_size, DECODER_HIDDEN_DIM]

        hidden_enc = torch.tanh(hidden_enc)

        return rnn_enc, hidden_enc
    
class Attention(nn.Module):
    def __init__(self, DECODER_HIDDEN_DIM, ENCODER_HIDDEN_DIM):
        super(Attention, self).__init__()

        self.fc1 = nn.Linear((ENCODER_HIDDEN_DIM*2) + DECODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)
        self.fc2 = nn.Linear(DECODER_HIDDEN_DIM, 1)
        
        
    def forward(self, encoder_output, decoder_hidden):
        # WRITE CODE HERE

        # encoder_output shape : [sequence_len, batch_size, ENCODER_HIDDEN_DIM*2]
        # decoder_hidden shape : [batch_size, DECODER_HIDDEN_DIM]

        # repeat decoder hidden state sequence_len Time's.       
        # input shape : [batch_size, DECODER_HIDDEN_DIM]
        decoder_hidden = torch.unsqueeze(decoder_hidden, 1)
        # output shape: [batch_size, 1, DECODER_HIDDEN_DIM]
        
        decoder_hidden = decoder_hidden.repeat(1, encoder_output.shape[0], 1)
        # [batch_size, sequence_len, DECODER_HIDDEN_DIM]
        
        encoder_output = encoder_output.permute(1, 0, 2)
        # encoder_output- shape : [batch_size, sequence_len, ENCODER_HIDDEN_DIM*2]
        
        # concatenate encoder's output and decoder's hidden state
        # and feed into a neural network layer
        concat = torch.cat((encoder_output, decoder_hidden), dim=2)
        # shape- [batch_size, sequence_len, (ENCODER_HIDDEN_DIM*2) + DECODER_HIDDEN_DIM], [batch_size, sequence_len, 3000]
        
        # input shape: [batch_size, sequence_len, (ENCODER_HIDDEN_DIM*2) + DECODER_HIDDEN_DIM]
        fc1 = self.fc1(concat)
        # output: [batch_size, sequence_len, DECODER_HIDDEN_DIM]
        
        fc1 = torch.tanh(fc1)
        # output: [batch_size, sequence_len, DECODER_HIDDEN_DIM]     
        
        fc2 = self.fc2(fc1)
        # output: [batch_size, sequence_len, 1]
        
        alpha = F.softmax(fc2, dim=1)
        # output: [batch_size, sequence_len, 1]
        
        # attention vector to take the weighted sum of the encoder hidden state.
        alpha=alpha.permute(0, 2, 1)
        # alpha shape:            [batch_size, 1, sequence_len]
        # encoder_output- shape : [batch_size, sequence_len, ENCODER_HIDDEN_DIM*2]
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        
        a=alpha@encoder_output     # multiplying all the words in each sequence, from 1..N, wher N=sequence len.
        # [batch_size, 1, sequence_len] * [batch_size, sequence_len, ENCODER_HIDDEN_DIM*2]
        # attention- shape : [batch_size, 1, ENCODER_HIDDEN_DIM*2]
        
        return a

class Decoder(nn.Module):
    def __init__(self, EMBEDDING_DIM, EMBEDDING_SIZE_SPANISH, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM):
        super(Decoder, self).__init__()
        
        self.embedding_dim = EMBEDDING_DIM
        self.embedding_size_spanish = EMBEDDING_SIZE_SPANISH
        self.hidden_dim = DECODER_HIDDEN_DIM
        self.encoder_hidden_dim = ENCODER_HIDDEN_DIM
        self.decoder_hidden_dim = DECODER_HIDDEN_DIM
        
        self.attention = Attention(self.decoder_hidden_dim, self.encoder_hidden_dim)
        
        self.embedding = nn.Embedding( self.embedding_size_spanish , self.embedding_dim)

        self.rnn = nn.RNN((self.encoder_hidden_dim*2)+self.embedding_dim, self.hidden_dim, bidirectional = False)

        self.linear = nn.Linear(self.embedding_dim+(self.encoder_hidden_dim*2)+self.decoder_hidden_dim, self.embedding_size_spanish)
        
    def forward(self, target_lang, encoder_output, decoder_hidden):
                
        # input shape: [BATCH_SIZE]
        embeds = self.embedding(target_lang)
        # embeds shape: [BATCH_SIZE, EMBEDDING_DIM]
        
        # attention layer - get the context vector from the attention layer.
        # encoder_output: [sequence_len, batch_size, ENCODER_HIDDEN_DIM*2]
        # decoder_hidden: [batch_size, DECODER_HIDDEN_DIM]
        a = self.attention(encoder_output, decoder_hidden)
        # attention shape: [BATCH_SIZE, 1, ENCODER_HIDDEN_DIM*2]
        
        ### concatenate convext vector, "embeds" and attention output, "a".
        
        # expand 1 dim for embeds
        #input shape: [BATCH_SIZE, EMBEDDING_DIM]
        embeds = embeds.unsqueeze(0)
        #output shape: [1, BATCH_SIZE, EMBEDDING_DIM]

        # change the shape of context vector.
        # input shape:  [BATCH_SIZE, 1, ENCODER_HIDDEN_DIM*2]
        a = a.permute(1, 0, 2)
        # output shape: [1, BATCH_SIZE, ENCODER_HIDDEN_DIM*2]
            
        x = torch.cat((embeds, a), dim=2)
        # output shape: [1, BATCH_SIZE, ENCODER_HIDDEN_DIM*2+EMBEDDING_DIM] i.e. [1, 16, 2030]
        
        # input shape: [1, BATCH_SIZE, ENCODER_HIDDEN_DIM*2+EMBEDDING_DIM]
        rnn_dec, decoder_hidden = self.rnn(x)
        #rnn_dec, decoder_hidden = self.rnn(x, decoder_hidden.unsqueeze(0))
        # rnn_dec shape: [1, BATCH_SIZE, DECODER_HIDDEN_DIM], *1 since it is not bidirectional.
        # hidden.shape: [1, BATCH_SIZE, DECODER_HIDDEN_DIM], hidden[0] = 1, since it is not bidirectional.
        
        # prediction
        # input to the nn: embeds, a, rnn_dec
        # embeds output shape:    [1, BATCH_SIZE, EMBEDDING_DIM]
        # attention output shape: [1, BATCH_SIZE, ENCODER_HIDDEN_DIM*2]
        # rnn_dec shape:          [1, BATCH_SIZE, DECODER_HIDDEN_DIM]
        
        c1=torch.cat((rnn_dec, a, embeds), dim=2)
        # c1 shape: [1, BATCH_SIZE, EMBEDDING_DIM+(ENCODER_HIDDEN_DIM*2)+DECODER_HIDDEN_DIM]
        
        pred = self.linear(c1).squeeze(0)
        # [BATCH_SIZE, EMBEDDING_SIZE_SPANISH]
        
        return pred, decoder_hidden.squeeze(0)


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
    
    # read the data and word_index
    DIR_path = "D:\MSc Data Science\Advanced Modules\[INF-DSAM1B] Advanced Machine Learning B\Deep learning for NLP\Project\Machine translation with attention"
    PROCESSED_DATA_PATH = "Processed Data"
    
    # for the time, valid data is being used.
    with open("D:\MSc Data Science\Advanced Modules\[INF-DSAM1B] Advanced Machine Learning B\Deep learning for NLP\Project\Machine translation with attention\Processed Data\\valid.pickle", "rb") as f:
        valid = pickle.load(f)
    #with open("spanish_tokenized_tensor.pickle", "rb") as f:
    #    spanish_tokenized_tensor = pickle.load(f)
    
    with open(os.path.join(DIR_path, PROCESSED_DATA_PATH, "word_to_index_eng.pickle"), "rb") as f:
        word_to_index_eng = pickle.load(f)
    
    with open(os.path.join(DIR_path, PROCESSED_DATA_PATH, "word_to_index_spanish.pickle"), "rb") as f:
         word_to_index_spanish = pickle.load(f)
    
    # For cuda.
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    BATCH_SIZE = 16
    MAX_SENTENCE_LENGTH = 50
    EMBEDDING_DIM = 30
    ENCODER_HIDDEN_DIM = 1000
    DECODER_HIDDEN_DIM = 1000
    EPOCHS = 10
    
    EMBEDDING_SIZE_ENGLISH = len(word_to_index_eng)
    EMBEDDING_SIZE_SPANISH = len(word_to_index_spanish)

    english_tokenized_tensor = valid[0]
    spanish_tokenized_tensor = valid[1]
    
    
    # encoder model
    # encoder model
    model = Encoder(EMBEDDING_DIM, EMBEDDING_SIZE_ENGLISH, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)
    model.to(device)
    
    model_attn = Attention(DECODER_HIDDEN_DIM, ENCODER_HIDDEN_DIM)
    model_attn.to(device)
    
    model_decoder = Decoder(EMBEDDING_DIM, EMBEDDING_SIZE_SPANISH, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM)
    model_decoder.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    loss_function = nn.CrossEntropyLoss()

    for epochs in range(EPOCHS):
        total_loss = 0
        model.to(device)
        for i in range(0, len(english_tokenized_tensor), BATCH_SIZE):
            english_batch = english_tokenized_tensor[i:i+BATCH_SIZE] 
            spanish_batch = spanish_tokenized_tensor[i:i+BATCH_SIZE] 
    
            en, es = padding(english_batch, spanish_batch) # shape : [sequence_length, batch_size]
    
            en = en.to(device)
            es = es.to(device)
    
            encoder_output, enc_hidden = model(en)
            
            decoder_hidden = enc_hidden    # need to change this somehow.
            
            #outputs = torch.zeros(MAX_SENTENCE_LENGTH, BATCH_SIZE, EMBEDDING_SIZE_SPANISH).to(device)
            outputs=[]
            spanish = es[0, :]
            
            # feed the target sentence, one by one word
            for i in range(MAX_SENTENCE_LENGTH):
                
                # at each loop, 1st word of every 16 batches (sentences) will be fed into the decoder.
                
                output, decoder_hidden = model_decoder(spanish, encoder_output, decoder_hidden)
                # output shape:         [BATCH_SIZE, EMBEDDING_SIZE_SPANISH]
                # decoder_hidden shape: [BATCH_SIZE, DECODER_HIDDEN_DIM]
                
                spanish = output.argmax(1)
                # spanish shape: [BATCH_SIZE]
                
                outputs.append(output)
                # outputs shape: [sequence_length, BATCH_SIZE, EMBEDDING_SIZE_SPANISH]
            
            outputs = torch.stack(outputs)
            # outputs shape: [sequence_length, BATCH_SIZE, EMBEDDING_SIZE_SPANISH]
            
            prediction = outputs.reshape(-1, EMBEDDING_SIZE_SPANISH)
            # shape: [sequence_length*BATCH_SIZE, EMBEDDING_SIZE_SPANISH]
            #print(prediction.shape)
            actual = es.reshape(-1)
            # shape: [sequence_length*BATCH_SIZE]
            
            l = loss_function(prediction, actual)
            total_loss += l
            optimizer.zero_grad() 
            l.backward() 
            optimizer.step()
        
        print('epoch: %d, loss: %.4f' % ((epochs+1), total_loss))
            