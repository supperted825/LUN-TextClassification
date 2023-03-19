import os
import nltk
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk import wordpunct_tokenize
from sklearn.model_selection import train_test_split

import nlpaug.flow as naf
import nlpaug.augmenter.word as naw


# ----- Read Data & Clean

# root = '/content/gdrive/MyDrive/DSML Coursework/CS4248 Project/raw_data/'
root = '/home/svu/e0425991/bert/'

df_train = pd.read_csv(os.path.join(root, './data/fulltrain.csv'), header=None)
df_train.columns = ['cls', 'text']

df_train = df_train.drop_duplicates()

df_train['num_tokens'] = df_train['text'].apply(lambda x: len(wordpunct_tokenize(x)))
df_train = df_train[df_train['num_tokens'] >= 10].reset_index(drop=True)
df_train = df_train.drop(columns='num_tokens')

train_idx, val_idx = train_test_split(
    np.arange(df_train.shape[0]),
    test_size = 0.2,
    shuffle = True,
    stratify = df_train['cls'].tolist(),
    random_state = 42)

df_val = df_train.loc[val_idx]
df_val.to_csv('./data/validation.csv', index=False, header=False)
print('Num Validation Samples:', df_val.shape[0])

# ----- Split Long Documents for More Samples

df_train = df_train.loc[train_idx].reset_index(drop=True)

def split_document(text):
    
    split_text, curr_text = [], []
    max_len, curr_len = 512, 0
    
    for sentence in nltk.sent_tokenize(text):
        tokens = nltk.wordpunct_tokenize(sentence)
        tokens_len = len(tokens)
        
        if curr_len < max_len:
            curr_text.extend(tokens)
            curr_len += tokens_len
        else:
            split_text.append(' '.join(curr_text))
            curr_text, curr_len = tokens, tokens_len
            
    if curr_len >= 10:
        split_text.append(' '.join(curr_text))
    
    return split_text

df_train['text'] = df_train['text'].apply(split_document)
df_train = df_train.explode('text')

df_train.to_csv('./data/train_split.csv')

# ----- Apply Augmentations to Training Data

aug = naf.Sequential([
    naw.RandomWordAug(action='swap'),
    naw.RandomWordAug(action='delete'),
    naw.RandomWordAug(action='crop'),
    naw.SynonymAug()
])

cls_names = { 0 : "satire", 1 : "hoax", 2 : "propaganda", 3 : "reliable"}

print('Num Train Samples before Augmentation:', df_train.shape[0])

texts = df_train['text'].tolist()
labels = df_train['cls'].tolist()

aug_data = []
problem_idxs = []

for idx, (text, label) in tqdm(enumerate(zip(texts, labels))):
    try:
        aug_text = aug.augment(text, n=3)
        for aug_sample in aug_text:
            aug_data.append((label, aug_sample))
    except:
        print(text, cls_names[label-1], idx)

aug_data = pd.DataFrame(aug_data, columns=['cls', 'text'])
aug_data.columns = ['cls', 'text']

df_train = pd.concat([df_train, aug_data]).reset_index(drop=True)
df_train.to_csv('./data/augmented_train.csv', index=False, header=False)

print('Num Samples after Augmentation:', df_train.shape[0])