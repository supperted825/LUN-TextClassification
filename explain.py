

import os
import time
import random
import joblib
import datetime
import argparse
import numpy as np
import pandas as pd

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from tqdm import trange, tqdm

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from src.models import TransformerClassifier
from src.utils import tokenize


root = None ### Path to your root folder
transformer = 'microsoft/deberta-v3-base'
load_model_path = 'deberta-v3-base.pth'

# ----- Load Data (No Need for Validation Set)

train_csv = './data/augmented_train.csv'

df_train = pd.read_csv(os.path.join(root, train_csv), header=None)
df_test  = pd.read_csv(os.path.join(root, './data/balancedtest.csv'), header=None)

df_train.columns = ['cls', 'text']
df_test.columns  = ['cls', 'text']

# ----- Get TF-IDF

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words=stopwords.words('english'),
    max_df=0.8,
    min_df=10,
    max_features=5096
)

train_tfidf = tfidf.fit_transform(df_train['text'].tolist())
test_tfidf  = tfidf.transform(df_test['text'].tolist())

train_tfidf = torch.from_numpy(train_tfidf.toarray())
test_tfidf  = torch.from_numpy(test_tfidf.toarray())

tfidf_feature_dims = train_tfidf.shape[-1]

# ----- Load Trained Model

model = TransformerClassifier(
    transformer = transformer,
    tfidf_dim = tfidf_feature_dims
)

model_state_dict = torch.load(load_model_path)['model_state_dict']
model.load_state_dict(model_state_dict)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(transformer)

# ----- Explainability Code for Test Set

test_tokens, test_att_masks, test_labels = tokenize(df_test, tokenizer)
test_data  = [test_tokens, test_att_masks, test_labels, test_tfidf]