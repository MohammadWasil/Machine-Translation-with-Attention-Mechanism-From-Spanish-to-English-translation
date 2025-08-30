import numpy as np
import os
import nltk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
nltk.download('punkt')

def read_data():
    print("Reading the file ...")
    DATA_PATH = "Data"
    
    # 'utf-8' removes b'' character string literal
    # splitlines() remove newline character
    with open(os.path.join(DATA_PATH, "europarl-v7.es-en.es"), "rb") as f:
        content_spanish = f.read().decode("utf-8").splitlines()
    
    with open(os.path.join(DATA_PATH, "europarl-v7.es-en.en"), "rb") as f:
        content_english = f.read().decode("utf-8").splitlines()
        
    return content_english, content_spanish