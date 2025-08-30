from __future__ import annotations

from pathlib import Path

from data_utils import read_data
from data_utils import sentence_preprocess
from data_utils import tokenize_en
from data_utils import tokenize_es
from torchtext.data import Field
from torchtext.data import Iterator
from torchtext.data import TabularDataset


def prepare_data():
    """Prepare and preprocess data if not already done."""
    if not (Path("train.csv").is_file() and Path("val.csv").is_file() and Path("test.csv").is_file()):
        content_english, content_spanish = read_data()
        sentence_preprocess(content_english, content_spanish)


def get_fields_and_datasets():
    """Create torchtext Fields and TabularDatasets."""
    sos_token = "<sos>"  # noqa: S105
    eos_token = "<eos>"  # noqa: S105

    source_field = Field(tokenize=tokenize_es, init_token=sos_token, eos_token=eos_token, lower=True)
    target_field = Field(tokenize=tokenize_en, init_token=sos_token, eos_token=eos_token, lower=True)
    data_fields = [("src", source_field), ("trg", target_field)]
    train_data, valid_data, test_data = TabularDataset.splits(
        path="", train="train.csv", validation="val.csv", test="test.csv", format="csv", fields=data_fields
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
