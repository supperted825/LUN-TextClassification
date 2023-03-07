# -*- coding: utf-8 -*-
"""
Training Script for Finetuning any transformer-based
classifier on the Labeled Unreliable News Dataset.
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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from models import TransformerClassifier

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('--transformer', default="xlnet-base-cased")
        self.parser.add_argument('--num_epochs', default=3, type=int)
        self.parser.add_argument('--batch_size', default=32, type=int)
        self.parser.add_argument('--reinit_layers', default=0, type=int)
        self.parser.add_argument('--freeze_backbone', action='store_true')
        self.parser.add_argument('--focal_loss', action='store_true')
        self.parser.add_argument('--downsample', action='store_true')
        
        self.parser.add_argument('--use_tfidf', action='store_true')
        self.parser.add_argument('--tfidf_features', default=2048, type=int)
        
    def parse(self, args=''):
        
        opt = self.parser.parse_args()
        args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
        
        print('Arguments:')
        for k, v in sorted(args.items()):
            print('  %s: %s' % (str(k), str(v)), flush=True)

        return opt

opt = opts().parse()

# ----- Read Data & Clean

# root = '/content/gdrive/MyDrive/DSML Coursework/CS4248 Project/raw_data/'
root = '/home/svu/e0425991/bert/'

df_train = pd.read_csv(os.path.join(root, './data/fulltrain.csv'), header=None)
df_test = pd.read_csv(os.path.join(root, './data/balancedtest.csv'), header=None)

df_train.columns = ['cls', 'text']
df_test.columns = ['cls', 'text']

# ----- Downsample to Induce Class Balance

if opt.downsample:

    df_train_new = pd.DataFrame([], columns=['cls', 'text'])
    min_cls_count = df_train['cls'].value_counts().min()

    for cls_ in range(1, 5):
        df_tmp = df_train[df_train['cls'] == cls_].sample(min_cls_count, replace=False)
        df_train_new = pd.concat([df_train_new, df_tmp])

    df_train = df_train_new.sample(frac=1).reset_index(drop=True)

# ----- Tokenize Training Data

tokenizer = AutoTokenizer.from_pretrained(opt.transformer, do_lower_case='uncased' in opt.transformer)
AutoModel.from_pretrained(opt.transformer)

def tokenize_sample(text, tokenizer):
    return tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens = True,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

def tokenize(df):
    token_id = []
    attention_masks = []

    for sample in tqdm(df['text'].values):
        encoding_dict = tokenize_sample(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])

    tokens = torch.cat(token_id, dim=0)
    att_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['cls'].astype(int).values) - 1
    
    return tokens, labels, att_masks
    
train_tokens, train_labels, train_att_masks = tokenize(df_train)
test_tokens, test_labels, test_att_masks = tokenize(df_test)

train_idx, val_idx = train_test_split(
    np.arange(len(train_labels)),
    test_size = 0.2,
    shuffle = True,
    stratify = train_labels)

# ----- Train and validation sets

train_data = [train_tokens[train_idx], train_att_masks[train_idx], train_labels[train_idx]]
val_data   = [train_tokens[val_idx], train_att_masks[val_idx], train_labels[val_idx]]
test_data  = [test_tokens, test_att_masks, test_labels]

if opt.use_tfidf:

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords.words('english'),
        max_df=0.8,
        min_df=10,
        max_features=5096
    )
    
    train_tfidf = tfidf.fit_transform(df_train.loc[train_idx, 'text'].tolist())
    val_tfidf   = tfidf.transform(df_train.loc[val_idx, 'text'].tolist())
    test_tfidf  = tfidf.transform(df_test['text'].tolist())
    
    train_tfidf = torch.from_numpy(train_tfidf.toarray())
    val_tfidf   = torch.from_numpy(val_tfidf.toarray())
    test_tfidf  = torch.from_numpy(test_tfidf.toarray())
    
    train_data += [train_tfidf]
    val_data   += [val_tfidf]
    test_data  += [test_tfidf]
    
    tfidf_feature_dims = train_tfidf.shape[-1]
else:
    tfidf_feature_dims = 0
    
train_set = TensorDataset(*train_data)
val_set = TensorDataset(*val_data)
test_set = TensorDataset(*test_data)

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

model = TransformerClassifier(
    transformer = opt.transformer,
    tfidf_dim = tfidf_feature_dims,
    reinit_layers = opt.reinit_layers,
    focal_alpha = focal_loss_weight,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 5e-5,
    eps = 1e-08
)

if opt.freeze_backbone:
    for param in model.backbone.parameters():
        param.requires_grad = False

# ----- Run on GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ----- Define Evaluation procedure for Validation & Test

def evaluate(dataloader):
    
    label_pred = []
    
    for batch in dataloader:
        
        batch = tuple(t.to(device) for t in batch)
        
        if opt.use_tfidf:
            b_input_ids, b_input_mask, _, b_tfidf = batch
        else:
            (b_input_ids, b_input_mask, _), b_tfidf = batch, None

        with torch.no_grad():
            eval_output = model(
                b_input_ids,
                tfidf_features = b_tfidf,
                attention_mask = b_input_mask
            )
            
        logits = eval_output.detach().cpu().numpy()
        b_label_pred = np.argmax(logits, axis=1).tolist()
        label_pred.extend(b_label_pred)
        
    return label_pred

# ----- Begin Training

for epoch in trange(opt.num_epochs, desc = 'Epoch'):
    
    tr_loss = nb_tr_examples = nb_tr_steps = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        batch = tuple(t.to(device) for t in batch)
        
        if opt.use_tfidf:
            b_input_ids, b_input_mask, b_labels, b_tfidf = batch
        else:
            (b_input_ids, b_input_mask, b_labels), b_tfidf = batch, None
        
        optimizer.zero_grad()
        
        loss = model(
            b_input_ids,
            tfidf_features = b_tfidf,
            attention_mask = b_input_mask, 
            labels = b_labels
        )
        
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        
        # if step % 50 == 0:

        #     label_pred = []
        #     for batch in test_dataloader:
        #         batch = tuple(t.to(device) for t in batch)
        #         b_input_ids, b_input_mask, _ = batch

        #         with torch.no_grad():
        #             eval_output = model(
        #                 b_input_ids, 
        #                 token_type_ids = None,
        #                 attention_mask = b_input_mask
        #             )
                    
        #         logits = eval_output.detach().cpu().numpy()
        #         b_label_pred = np.argmax(logits, axis=1).tolist()
        #         label_pred.extend(b_label_pred)
            
        #     print()
        #     print(classification_report(test_labels, label_pred))
        #     print(flush=True)

    model.eval()
    
    print(f'Epoch {epoch}')

    print('Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    
    val_label_pred = evaluate(validation_dataloader)
    
    print('Validation Set Classification Report\n')
    print(classification_report(train_labels[val_idx], val_label_pred))
    print('Micro F1 : {:.4f}'.format(f1_score(train_labels[val_idx], val_label_pred, average='micro')))

    test_label_pred = evaluate(test_dataloader)
    
    print('Test Set Classification Report\n')
    print(classification_report(test_labels, test_label_pred))
    print('Micro F1 : {:.4f}'.format(f1_score(test_labels, test_label_pred, average='micro')))
    print('\n', classification_report(test_labels, test_label_pred, output_dict=True), flush=True)

def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(root, 'BERT.pth'))

save(model, optimizer)