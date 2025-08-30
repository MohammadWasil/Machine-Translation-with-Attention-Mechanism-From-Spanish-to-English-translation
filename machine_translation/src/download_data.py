from __future__ import annotations

import os
import tarfile


def Download_and_extract():
    os.system("wget https://www.statmt.org/europarl/v7/es-en.tgz")

    # open file
    file_path = Path("es-en.tgz")
    file = tarfile.open(file_path)
    # extracting file
    file.extractall("./Data")
    file.close()
