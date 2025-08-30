from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path


def download_and_extract():
    subprocess.run(
        ["wget", "https://www.statmt.org/europarl/v7/es-en.tgz"],  # noqa: S607
        check=True,
    )

    # open file
    file_path = Path("es-en.tgz")
    file = tarfile.open(file_path)  # noqa: SIM115
    # extracting file
    file.extractall("./Data")  # noqa: S202
    file.close()
