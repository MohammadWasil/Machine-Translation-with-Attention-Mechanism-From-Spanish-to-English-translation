from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from machine_translation.src.dataset import get_fields_and_datasets
from machine_translation.src.dataset import get_iterators
from machine_translation.src.dataset import prepare_data
from machine_translation.src.download_data import Download_and_extract
from machine_translation.src.inference import calculate_bleu
from machine_translation.src.model import Model
from machine_translation.src.train import Trainer

"""
Pytroch version :  1.7.1+cu101
torchtext version: 0.8.0
spacy version:     3.1.1
"""


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Experiment Args")

    parser.add_argument(
        "--RUN_MODE",
        dest="RUN_MODE",
        choices=["train_val", "test", "bleu"],
        help="{train_val, test, bleu}",
        type=str,
        required=True,
    )

    parser.add_argument("--CPU", dest="CPU", help="use CPU instead of GPU", action="store_true")

    parser.add_argument("--MODEL", dest="MODEL", help="upload trained model", type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path("./config.yml")
    with config_path.open() as f:
        config = yaml.safe_load(f)

    data_dir = Path("Data")
    if not (data_dir / "europarl-v7.es-en.en").is_file() or not (data_dir / "europarl-v7.es-en.es").is_file():
        Download_and_extract()

    prepare_data()
    source_field, target_field, train_data, valid_data, test_data = get_fields_and_datasets()
    train_iterator, valid_iterator, test_iterator = get_iterators(
        train_data,
        valid_data,
        test_data,
        config["batch_size"],
        torch.device("cuda" if torch.cuda.is_available() and not args.CPU else "cpu"),
    )

    model = Model(
        len(source_field.vocab),
        len(target_field.vocab),
        config["EMBEDDING_DIM"],
        config["ENCODER_HIDDEN_DIM"],
        config["DECODER_HIDDEN_DIM"],
        torch.device("cuda" if torch.cuda.is_available() and not args.CPU else "cpu"),
    )
    trainer = Trainer(
        model,
        config,
        torch.device("cuda" if torch.cuda.is_available() and not args.CPU else "cpu"),
        len(target_field.vocab),
    )

    if args.RUN_MODE == "train_val":
        trainer.train(train_iterator, valid_iterator)
    elif args.RUN_MODE == "test":
        model_number = args.MODEL
        state = torch.load(f"model_{model_number}.pickle")
        model.load_state_dict(state["state_dict"])
        test_loss = trainer.evaluate(test_iterator)
        print(f"Test Loss: {test_loss}")  # noqa: T201
    elif args.RUN_MODE == "bleu":
        model_number = args.MODEL
        state = torch.load(f"model_{model_number}.pickle")
        model.load_state_dict(state["state_dict"])
        score = calculate_bleu(
            model,
            test_data,
            source_field,
            target_field,
            torch.device("cuda" if torch.cuda.is_available() and not args.CPU else "cpu"),
        )
        print(f"BLEU score = {score * 100.00}")  # noqa: T201


if __name__ == "__main__":
    main()
