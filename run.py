import os
import argparse
import yaml

import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.data.metrics import bleu_score

from mymodel import Model
from data_utils import read_data, sentence_preprocess, tokenize_en, tokenize_es
from download_data import Download_and_extract

'''
Pytroch version :  1.7.1+cu101
torchtext version: 0.8.0
spacy version:     3.1.1
'''

try:
    from torchtext.data import Field, TabularDataset, Iterator
    print('Torchtext imported successfully')
except ImportError:
    print('\n Torchtext is not imported. Trying to import some other way...')
    try:
        from torchtext.legacy.data import Field, TabularDataset, Iterator
        print("Torchtext imported succesfully")
    except ImportError:
        print("\n Torchtext is not installed. Trying to install package...")
        try:
            os.system('pip install torchtext')
        except ImportError:
            print('Not able to install TorchText!')

try:
    os.system('pip install -U spacy==3.1.1')
    import spacy    
    print('Spacy 3.1.1 imported successfully')
except ImportError:
    print('\n  Spacy is not installed. Trying to install package...')
    try:
        os.system('pip install -U spacy==3.1.1')
        print('Spacy 3.1.1 version installed')
    except ImportError:
        print('Not able to install Spacy!')


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Experiment Args')

    parser.add_argument(
        '--RUN_MODE', dest='RUN_MODE',
        choices=['train_val', 'test', 'bleu'],
        help='{train_val, test, bleu}',
        type=str, required=True
    )

    parser.add_argument(
        '--CPU', dest='CPU',
        help='use CPU instead of GPU',
        action='store_true'
    )

    parser.add_argument(
        '--MODEL', dest='MODEL',
        help='upload trained model',
        type=int
    )

    args = parser.parse_args()
    return args

class MainExec(object):
    def __init__(self, args, configs):
        self.args = args
        self.cfgs = configs

        if self.args.CPU:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )  # for failsafe
        if self.args.RUN_MODE == 'test' or self.args.RUN_MODE =='bleu':
            if self.args.MODEL == None:
                raise Exception('Add a model number you need to evaluate, e.g Model_8.pickle, then pass 8 as an argument')

    def init_weights(self, m):
        # initialize weights of the model with xavier_uniform
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    def translate_sentence(self, sentence, src_field, trg_field, model, device, max_len = 50):
        self.model.eval()
            
        if isinstance(sentence, str):
            nlp = spacy.load('es')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]
    
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
            
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]
        
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
            
        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
        
        for i in range(max_len):
    
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                    
            with torch.no_grad():
                output, hidden = self.model.decoder(trg_tensor, hidden, encoder_outputs)
                
            pred_token = output.argmax(1).item()
            
            trg_indexes.append(pred_token)
    
            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break
        
        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        
        return trg_tokens[1:]
    
    
    def evaluate(self, model, iterator, loss_function):
    
        model.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
        
            for i, batch in enumerate(iterator):
    
                source = batch.src
                target = batch.trg
    
                # feed the source and target sentence into the model to get prediction
                output = model(source, target, 0)
                
                prediction = output[1:].reshape(-1, self.EMBEDDING_SIZE_ENGLISH)
                # shape: [sequence_len_target-1*BATCH_SIZE, EMBEDDING_SIZE_ENGLISH]        
                
                actual = target[1:].reshape(-1)
                # shape: [sequence_len_target-1*BATCH_SIZE]

                loss = loss_function(prediction, actual)

                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def train(self, train_iterator, valid_iterator):
        
        optimizer = Adam(self.model.parameters(), self.cfgs['lr'])
        loss_function = nn.CrossEntropyLoss()
        
        for epoch in range(self.cfgs["epochs"]):
            
            self.model.train()
            self.model.to(self.device)
            train_loss = 0
            for i, batch in enumerate(train_iterator):
            
                source = batch.src
                target = batch.trg
                # src shape: [sequence_len_source, BATCH_SIZE]
                # trg shape: [sequence_len_target, BATCH_SIZE]
        
                optimizer.zero_grad()
                
                # feed the source and target sentence into the model to get prediction
                output = self.model(source, target)
                #output shape : [sequence_len_target, BATCH_SIZE, EMBEDDING_SIZE_ENGLISH]
        
                prediction = output[1:].reshape(-1, self.EMBEDDING_SIZE_ENGLISH)
                # prediction shape: [sequence_len_target-1*BATCH_SIZE, EMBEDDING_SIZE_ENGLISH]        
        
                actual = target[1:].reshape(-1)
                # actual shape: [sequence_len_target-1*BATCH_SIZE]
                
                loss = loss_function(prediction, actual)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                
                optimizer.step()
                
                train_loss += loss.item()
            train_loss = train_loss / len(train_iterator)
            
            valid_loss = self.evaluate(self.model, valid_iterator, loss_function)

            # save the model at every epoch.
            state = {
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict()
            }
            torch.save(state, f'model_{epoch+1}.pickle')

            print(f'Epoch: {epoch+1} | Train Loss: {train_loss} | Val. Loss: {valid_loss}')   
    
    def run(self, run_mode):
        MAX_SENTENCE_LENGTH = 50
        
        # if the process file already exists, then there is no need to process and split the data again.
        if((os.path.isfile('train.csv') == False) or (os.path.isfile('val.csv') == False) or (os.path.isfile('test.csv') == False)):
            
            content_english, content_spanish = read_data()
            sentence_preprocess(content_english, content_spanish)
            
        # Tokenizing
        print("Tokenizing ...")
        source_field = Field(tokenize = tokenize_es, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    
        target_field = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
        
        data_fields = [('src', source_field), ('trg', target_field)]
            
        train_data, valid_data, test_data = TabularDataset.splits(path='',
                                                train='train.csv', validation='val.csv', test='test.csv',
                                                format = 'csv',
                                                fields = data_fields)
        
        source_field.build_vocab(train_data, min_freq = 2)
        target_field.build_vocab(train_data, min_freq = 2)
        
        print("Creating Data Iterator ...")
        train_iterator = Iterator(train_data, batch_size=self.cfgs["batch_size"], device=self.device, shuffle=False)
        valid_iterator = Iterator(valid_data, batch_size=self.cfgs["batch_size"], device=self.device, shuffle=False)
        test_iterator  = Iterator(test_data, batch_size=self.cfgs["batch_size"], device=self.device, shuffle=False)
        
        print("Creating Model ...")
        self.EMBEDDING_SIZE_SPANISH = len(source_field.vocab)
        self.EMBEDDING_SIZE_ENGLISH = len(target_field.vocab)
        
        EMBEDDING_DIM      = self.cfgs["EMBEDDING_DIM"]
        ENCODER_HIDDEN_DIM = self.cfgs["ENCODER_HIDDEN_DIM"]        
        DECODER_HIDDEN_DIM = self.cfgs["DECODER_HIDDEN_DIM"]
        
        self.model = Model(self.EMBEDDING_SIZE_SPANISH, self.EMBEDDING_SIZE_ENGLISH, EMBEDDING_DIM, ENCODER_HIDDEN_DIM, DECODER_HIDDEN_DIM, self.device).to(self.device)    
        self.model.apply(self.init_weights)
        
        if run_mode == 'train_val':
            print('Training...')
            self.train(train_iterator, valid_iterator)
        elif run_mode == 'test':
            print('Testing ...')
            
            model_number = args.MODEL
            # first load the model, and then evaluate.
            # model_number - input from the argparse
            state = torch.load(f'model_{model_number}.pickle')
            self.model.load_state_dict(state['state_dict'])

            loss_function = nn.CrossEntropyLoss()
            test_loss = self.evaluate(self.model, test_iterator, loss_function)
            print(f'Test Loss: {test_loss} ')
        elif run_mode == 'bleu':
            
            model_number = args.MODEL
            # first load the model, and then evaluate.
            # model_number - input from the argparse
            state = torch.load(f'model_{model_number}.pickle')
            self.model.load_state_dict(state['state_dict'])
            
            trgs = []
            pred_trgs = []
            
            for datum in test_data:
                
                source = vars(datum)['src']
                target = vars(datum)['trg']
                
                pred_trg = self.translate_sentence(source, source_field, target_field, self.model, self.device, max_len=MAX_SENTENCE_LENGTH)
                
                #cut off <eos> token
                pred_trg = pred_trg[:-1]
                
                pred_trgs.append(pred_trg)
                trgs.append([target])
            
            score = bleu_score(pred_trgs, trgs)
            print(f'BLEU score = {score*100.00}')

if __name__ == "__main__":
    args = parse_args()

    # download and extract the data. This will run only once. 
    if((os.path.isfile(os.path.join('Data', 'europarl-v7.es-en.en')) == False) or (os.path.isfile(os.path.join('Data', 'europarl-v7.es-en.es')) == False) ):
        Download_and_extract()
    

    with open('./config.yml', 'r') as f:
        model_config = yaml.safe_load(f)

    exec = MainExec(args, model_config)
    exec.run(args.RUN_MODE)