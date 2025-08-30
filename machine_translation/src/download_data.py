import os
import tarfile

def Download_and_extract():
  os.system('wget https://www.statmt.org/europarl/v7/es-en.tgz')
    
  # open file
  file = tarfile.open('es-en.tgz')
  # extracting file
  file.extractall('./Data')
  file.close()