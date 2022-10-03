# -*- coding: utf-8 -*-
# @File       : encode text.py
# @Author     : Yuchen Chai
# @Date       : 2022/9/14 9:07
# @Description:


import os
import torch
import pandas as pd
from utils import encode_text

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Please adjust the following parameters before executing the code
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# Please specify the location you put the source data
DIR_INPUT = ''

# Please specify the location you store the output data
DIR_OUTPUT = ''

# This is a default pretrained model we select. Feel free to change to others
PARA_MODEL_NAME = "stsb-xlm-r-multilingual"

# According to our statistics, 99% posts have a length less than 52
PARA_MAX_SEQ_LENGTH = 52

# The size of the batch
PARA_BATCH_SIZE = 100

# Decide the mode of the code
# func == 1: encode training dataset (sentiment140)
# func == 2: encode validation dataset (european 15 languages)
func= 1

if func == 1:
    print("Encoding training dataset")

    file_name = f"processed_training_sentiment140.csv"
    embedding_name = f"embeddings_training_{PARA_MODEL_NAME}_{PARA_MAX_SEQ_LENGTH}.pkl"

    print("Reading in training data")
    df = pd.read_csv(os.path.join(DIR_INPUT, 'training.1600000.processed.noemoticon.csv'), encoding='latin',header=None, usecols=[0, 5])
    df.columns = ['label', 'text']
    embedding, encoder, processed_text = encode_text(df, p_max_len=PARA_MAX_SEQ_LENGTH)

    torch.save(embedding, os.path.join(DIR_OUTPUT, f'out\\{embedding_name}'), pickle_protocol=5)
    processed_text.to_csv(os.path.join(DIR_OUTPUT, f"out\\{file_name}"))


elif func == 2:
    print("Encoding validation dataset")

    file_name = f"processed_validation_multilingual.csv"
    embedding_name = f"embeddings_validation_multilingual_{PARA_MODEL_NAME}_{PARA_MAX_SEQ_LENGTH}.pkl"

    print("Reading in training data")
    df = pd.read_csv(os.path.join(DIR_INPUT, 'validation_multilingual.csv'))
    df.columns = ['label', 'text']
    embedding, encoder, processed_text = encode_text(df, drop_duplicate=True, p_max_len=PARA_MAX_SEQ_LENGTH)

    torch.save(embedding, os.path.join(DIR_OUTPUT, f'out\\{embedding_name}'), pickle_protocol=5)
