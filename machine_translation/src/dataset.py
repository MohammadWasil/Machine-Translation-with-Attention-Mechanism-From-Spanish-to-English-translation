import os
from torchtext.data import Field, TabularDataset, Iterator
from data_utils import read_data, sentence_preprocess, tokenize_en, tokenize_es

def prepare_data():
        """Prepare and preprocess data if not already done."""
        if not (os.path.isfile('train.csv') and os.path.isfile('val.csv') and os.path.isfile('test.csv')):
                content_english, content_spanish = read_data()
                sentence_preprocess(content_english, content_spanish)

def get_fields_and_datasets():
        """Create torchtext Fields and TabularDatasets."""
        source_field = Field(tokenize=tokenize_es, init_token='<sos>', eos_token='<eos>', lower=True)
        target_field = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
        data_fields = [('src', source_field), ('trg', target_field)]
        train_data, valid_data, test_data = TabularDataset.splits(
                path='', train='train.csv', validation='val.csv', test='test.csv',
                format='csv', fields=data_fields
        )
        source_field.build_vocab(train_data, min_freq=2)
        target_field.build_vocab(train_data, min_freq=2)
        return source_field, target_field, train_data, valid_data, test_data

def get_iterators(train_data, valid_data, test_data, batch_size, device):
        """Create iterators for train, validation, and test sets."""
        train_iterator = Iterator(train_data, batch_size=batch_size, device=device, shuffle=False)
        valid_iterator = Iterator(valid_data, batch_size=batch_size, device=device, shuffle=False)
        test_iterator = Iterator(test_data, batch_size=batch_size, device=device, shuffle=False)
        return train_iterator, valid_iterator, test_iterator
