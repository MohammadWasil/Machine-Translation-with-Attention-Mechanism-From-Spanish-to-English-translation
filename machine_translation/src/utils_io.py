from __future__ import annotations

from pathlib import Path

import nltk

nltk.download("punkt")


def read_data():
    print("Reading the file ...")  # noqa: T201
    data_path = "Data"

    # 'utf-8' removes b'' character string literal
    # splitlines() remove newline character
    es_path = Path(data_path) / "europarl-v7.es-en.es"
    en_path = Path(data_path) / "europarl-v7.es-en.en"
    with es_path.open("rb") as f:
        content_spanish = f.read().decode("utf-8").splitlines()

    with en_path.open("rb") as f:
        content_english = f.read().decode("utf-8").splitlines()

    return content_english, content_spanish
