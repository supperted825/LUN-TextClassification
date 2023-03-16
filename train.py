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

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from tqdm import trange, tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support as score

from src.models import TransformerClassifier
from src.learning_rate import LayerwiseLR, LowerBackboneLR
from src.utils import tokenize


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--seed', default=42, type=int)
        self.parser.add_argument('--transformer', default="xlnet-base-cased")
        
        self.parser.add_argument('--num_epochs', default=5, type=int)
        self.parser.add_argument('--batch_size', default=16, type=int)
        self.parser.add_argument('--lr', default=5e-5, type=float)
        
        self.parser.add_argument('--focal_loss', action='store_true')
        self.parser.add_argument('--use_tfidf', action='store_true')
        self.parser.add_argument('--tfidf_features', default=5096, type=int)
        self.parser.add_argument('--use_augment', action='store_true')
        
        self.parser.add_argument('--layerwise_lrdecay', action='store_true')
        self.parser.add_argument('--lower_backbone_lr', action='store_true')
        self.parser.add_argument('--decay_factor', default=0.9, type=float)
        self.parser.add_argument('--reinit_layers', default=0, type=int)
        self.parser.add_argument('--freeze_backbone', action='store_true')
        self.parser.add_argument('--unfreeze_layers', default=0, type=int)
        
    def parse(self, args=''):
        
        opt = self.parser.parse_args()
        
        if opt.unfreeze_layers > 0:
            opt.freeze_backbone = True
        
        print('Arguments:')
        args = dict((name, getattr(opt, name)) for name in dir(opt) if not name.startswith('_'))
            
        with open(f'./logs/{opt.exp_id}_opt.txt', 'w+', newline ='') as file:
            for k, v in sorted(args.items()):
                print('  %s: %s' % (str(k), str(v)), flush=True)
                file.write('  %s: %s\n' % (str(k), str(v)))

        return opt


opt = opts().parse()
torch.manual_seed(opt.seed)

# ----- Read Data

# root = '/content/gdrive/MyDrive/DSML Coursework/CS4248 Project/raw_data/'
root = '/home/svu/e0425991/bert/'

if opt.use_augment:
    train_csv = './data/augmented_train.csv'
else:
    train_csv = './data/train.csv'

df_train = pd.read_csv(os.path.join(root, train_csv), header=None)
df_val   = pd.read_csv(os.path.join(root, './data/validation.csv'), header=None)
df_test  = pd.read_csv(os.path.join(root, './data/balancedtest.csv'), header=None)

df_train.columns = ['cls', 'text']
df_val.columns   = ['cls', 'text']
df_test.columns  = ['cls', 'text']

# ----- Tokenize & Prepare Data

tokenizer = AutoTokenizer.from_pretrained(opt.transformer, do_lower_case='uncased' in opt.transformer)
AutoModel.from_pretrained(opt.transformer)

train_tokens, train_att_masks, train_labels = tokenize(df_train, tokenizer)
val_tokens, val_att_masks, val_labels = tokenize(df_val, tokenizer)
test_tokens, test_att_masks, test_labels = tokenize(df_test, tokenizer)

train_data = [train_tokens, train_att_masks, train_labels]
val_data   = [val_tokens, val_att_masks, val_labels ]
test_data  = [test_tokens, test_att_masks, test_labels]

if opt.use_tfidf:

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords.words('english'),
        max_df=0.8,
        min_df=10,
        max_features=5096
    )
    
    train_tfidf = tfidf.fit_transform(df_train['text'].tolist())
    val_tfidf   = tfidf.transform(df_val['text'].tolist())
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

if opt.layerwise_lrdecay:
    optimizer = LayerwiseLR(model, opt.lr, opt.decay_factor)
elif opt.lower_backbone_lr:
    optimizer = LowerBackboneLR(model, opt.lr)
else:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = opt.lr,
        eps = 1e-08
    )

# ----- Configure Backbone Freezing

if opt.freeze_backbone:
    for param in model.backbone.parameters():
        param.requires_grad = False
        
if opt.unfreeze_layers > 0:
    match = [f'encoder.layer.{i}.' for i in range(12 - opt.unfreeze_layers, 12)]
    match = match + ['encoder.rel_embeddings.weight', 'encoder.LayerNorm', 'pooler']
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(m) for m in match):
            param.requires_grad = True
            continue

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

# ----- Set up Logging to CSV

idx = ['precision', 'recall', 'f1', 'support']
col = list(range(4)) + ['macro']
items = [i + '_' + str(c) for i in idx for c in col] + ['accuracy', 'micro_f1']
items_ordered = [i + '_' + str(c) for c in col for i in idx] + ['accuracy', 'micro_f1']
results_df = pd.DataFrame(columns=items)

# ----- Model Saving

best_f1 = 0.0
save_name = opt.transformer.split('/')[-1]

def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(root, f'{save_name}.pth'))

# ----- Begin Training

for epoch in trange(opt.num_epochs, desc = 'Epoch'):
    
    if epoch == 5:
        break
    
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

    model.eval()
    
    val_label_pred = evaluate(validation_dataloader)
    test_label_pred = evaluate(test_dataloader)
    
    print(f'Epoch {epoch + 1} - {opt.exp_id}')
    print('Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    
    print('Validation Set Classification Report\n')
    print(classification_report(val_labels, val_label_pred))

    print('Test Set Classification Report\n')
    print(classification_report(test_labels, test_label_pred), flush=True)
    
    micro_f1 = f1_score(test_labels, test_label_pred, average='micro')
    macro_f1 = f1_score(test_labels, test_label_pred, average='macro')
    epoch_report = classification_report(test_labels, test_label_pred, output_dict=True)
    
    res = pd.DataFrame(epoch_report)
    acc = res['accuracy'].mean()
    res = res.drop(columns=['accuracy', 'weighted avg'])
    results_df.loc[epoch] = res.to_numpy().flatten().tolist() + [acc, micro_f1]
    
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        save(model, optimizer)

results_df = results_df[items_ordered].round(3)
results_df.to_csv(f'./logs/{opt.exp_id}.csv')