# -*- coding: utf-8 -*-
"""
Training Script for Conducting Ablative Studies for
Text Classification on the Labeled Unreliable News Dataset.
"""

import os
import time
import random
import joblib
import datetime
import argparse
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from tqdm import trange, tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from models import MLPClassifier

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--exp_id', default='log-reg')
        self.parser.add_argument('--num_epochs', default=20, type=int)
        self.parser.add_argument('--batch_size', default=32, type=int)
        self.parser.add_argument('--focal_loss', action='store_true')
        self.parser.add_argument('--tfidf_features', default=5096, type=int)
        
    def parse(self, args=''):
        
        opt = self.parser.parse_args()
        args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
        
        print('Arguments:')
        for k, v in sorted(args.items()):
            print('  %s: %s' % (str(k), str(v)), flush=True)
            
        with open(f'./logs/{opt.exp_id}_opt.txt', 'w+', newline ='') as file:
            args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
            for k, v in sorted(args.items()):
                file.write('  %s: %s\n' % (str(k), str(v)))

        return opt

opt = opts().parse()

# ----- Read Data & Clean

# root = '/content/gdrive/MyDrive/DSML Coursework/CS4248 Project/raw_data/'
root = '/home/svu/e0425991/bert/'

df_train = pd.read_csv(os.path.join(root, './data/fulltrain.csv'), header=None)
df_test = pd.read_csv(os.path.join(root, './data/balancedtest.csv'), header=None)

df_train.columns = ['cls', 'text']
df_test.columns = ['cls', 'text']

df_train['cls'] = df_train['cls'] - 1
df_test['cls'] = df_test['cls'] - 1

# ----- Train Test Split

train_idx, val_idx = train_test_split(
    np.arange(len(df_train)),
    test_size = 0.2,
    shuffle = True,
    stratify = df_train['cls'])

# ----- Train and validation sets

tfidf = TfidfVectorizer(
    ngram_range = (1, 2),
    stop_words = stopwords.words('english'),
    max_df = 0.8,
    min_df = 10,
    max_features=opt.tfidf_features
)

train_tfidf = tfidf.fit_transform(df_train.loc[train_idx, 'text'].tolist())
val_tfidf   = tfidf.transform(df_train.loc[val_idx, 'text'].tolist())
test_tfidf  = tfidf.transform(df_test['text'].tolist())

train_tfidf = torch.from_numpy(train_tfidf.toarray())
val_tfidf   = torch.from_numpy(val_tfidf.toarray())
test_tfidf  = torch.from_numpy(test_tfidf.toarray())

train_labels = torch.from_numpy(df_train.loc[train_idx,'cls'].to_numpy())
val_labels   = torch.from_numpy(df_train.loc[val_idx, 'cls'].to_numpy())
test_labels  = torch.from_numpy(df_test['cls'].to_numpy())
    
train_set = TensorDataset(train_tfidf, train_labels)
val_set   = TensorDataset(val_tfidf, val_labels)
test_set  = TensorDataset(test_tfidf, test_labels)

tfidf_feature_dims = train_tfidf.shape[-1]

# ----- Stratify Batches for Train Loader

y_train = train_labels[train_idx].numpy()
class_sample_count = [(y_train == t).sum() for t in range(4)]

weight = 1. / np.array(class_sample_count)
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)

sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

# ----- Prepare DataLoaders

train_dataloader = DataLoader(
    train_set,
    shuffle = True,
    batch_size = opt.batch_size
)

validation_dataloader = DataLoader(
    val_set,
    shuffle = False,
    batch_size = opt.batch_size
)

test_dataloader = DataLoader(test_set, batch_size = opt.batch_size)

# ----- Initialise Model & Optimizer

focal_loss_weight = torch.Tensor([0.5, 0.5, 1, 0.5]).float() if opt.focal_loss else None

model = MLPClassifier(
    tfidf_dim = tfidf_feature_dims,
    fc_dim = 768
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 5e-5,
    eps = 1e-08
)

# ----- Run on GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ----- Define Evaluation procedure for Validation & Test

def evaluate(dataloader):
    
    label_pred = []
    
    for batch in dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        b_tfidf, _ = batch

        with torch.no_grad():
            eval_output = model(b_tfidf, labels=None)

        logits = eval_output.detach().cpu().numpy()
        b_label_pred = np.argmax(logits, axis=1).tolist()
        label_pred.extend(b_label_pred)
        
    return label_pred

# ----- Set up Logging to CSV

idx = ['precision', 'recall', 'f1', 'support']
col = list(range(4)) + ['macro']
items = [i + '_' + str(c) for i in idx for c in col] + ['accuracy', 'micro_f1']
items_ordered = [i + '_' + str(c) for c in col for i in idx] + ['accuracy', 'micro_f1']
results_df = pd.DataFrame(columns=items)

# ----- Begin Training

for epoch in trange(opt.num_epochs, desc = 'Epoch'):
    
    tr_loss = nb_tr_steps = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        batch = tuple(t.to(device) for t in batch)
        b_tfidf, b_labels = batch
        
        optimizer.zero_grad()
        
        loss = model(b_tfidf, b_labels)
        
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_steps += 1

    model.eval()
    
    val_label_pred = evaluate(validation_dataloader)
    test_label_pred = evaluate(test_dataloader)
    
    print(f'Epoch {epoch} - {opt.exp_id}')
    print('Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    
    print('Validation Set Classification Report\n')
    print(classification_report(train_labels[val_idx], val_label_pred))

    print('Test Set Classification Report\n')
    print(classification_report(test_labels, test_label_pred), flush=True)
    
    micro_f1 = f1_score(test_labels, test_label_pred, average='micro')
    epoch_report = classification_report(test_labels, test_label_pred, output_dict=True)
    
    res = pd.DataFrame(epoch_report)
    acc = res['accuracy'].mean()
    res = res.drop(columns=['accuracy', 'weighted avg'])
    results_df.loc[epoch] = res.to_numpy().flatten().tolist() + [acc, micro_f1]


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(root, 'BERT.pth'))


save(model, optimizer)
results_df = results_df[items_ordered]
results_df.to_csv(f'./logs/{opt.exp_id}.csv')