# -*- coding: utf-8 -*-
# @File       : cpu method.py
# @Author     : Yuchen Chai
# @Date       : 2022/9/14 9:33
# @Description:

import os
import pandas as pd
import json
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# Please adjust the following parameters before executing the code
# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

# Please specify the location you put the source data
DIR_INPUT_TEXT = ''
DIR_INPUT_EMBEDDING = ''

# Please specify the location you store the output data
DIR_OUTPUT = ''


def read_in_data(text_path, embed_path):
    df = pd.read_csv(text_path)

    print("Preparing training and test sets")
    with open(os.path.join(text_path, "train_ids.txt"), 'r') as fp:
        train_ids = json.load(fp)
    with open(os.path.join(text_path, "test_ids.txt"), 'r') as fp:
        test_ids = json.load(fp)

    embeddings = torch.load(embed_path)

    return df, train_ids, test_ids, embeddings


# Read in training dataset
df, train_ids, test_ids, embeddings = read_in_data(
    text_path=os.path.join(DIR_INPUT_TEXT, 'processed_training_sentiment140.csv'),
    embed_path=os.path.join(DIR_INPUT_EMBEDDING, 'embeddings_training_sentiment140_stsb-xlm-r-multilingual_52.pkl'))

# Create and Train Model
train_df = df.loc[train_ids, :]
X_train = embeddings[train_ids, :]
y_train = train_df['label'].values
test_df = df.loc[test_ids, :]
X_test = embeddings[test_ids, :]
y_test = test_df['label'].values


pca = PCA(n_components = 100, random_state=123)
logreg = LogisticRegression(random_state=123, solver='lbfgs', max_iter=100, C=1, penalty="l2")
pipe = Pipeline([('pca', pca), ('logreg', logreg)])
clf = pipe.fit(X_train, y_train)
torch.save(clf, os.path.join(DIR_OUTPUT, "cpu_clf.pkl"))
