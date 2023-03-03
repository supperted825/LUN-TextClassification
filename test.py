import os
import time
import random
import joblib
import datetime
import numpy as np
import pandas as pd

from tqdm import trange, tqdm
from tabulate import tabulate

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

root = '/home/svu/e0425991/bert/'



df_test = pd.read_csv(os.path.join(root, 'balancedtest.csv'), header=None)
df_test.columns = ['cls', 'text']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenize(text, tokenizer):
    return tokenizer.encode_plus(text,
                                 add_special_tokens = True,
                                 padding = 'max_length',
                                 truncation = True,
                                 return_attention_mask = True,
                                 return_tensors = 'pt')

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 4,
    output_attentions = False,
    output_hidden_states = False,
)

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, eps = 1e-08)

checkpoint = torch.load(os.path.join(root, 'BERT.pth'), map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

token_id = []
attention_masks = []

for sample in tqdm(df_test['text'].values):
    encoding_dict = tokenize(sample, tokenizer)
    token_id.append(encoding_dict['input_ids'])
    attention_masks.append(encoding_dict['attention_mask'])

tokens = torch.cat(token_id, dim=0)
att_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df_test['cls'].values) - 1

test_set = TensorDataset(tokens, att_masks, labels)

test_dataloader = DataLoader(test_set, batch_size = 32)

label_pred = []

for batch in test_dataloader:

    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, _ = batch

    with torch.no_grad():
        eval_output = model(b_input_ids, 
                            token_type_ids = None, 
                            attention_mask = b_input_mask)
        
    logits = eval_output.logits.detach().cpu().numpy()
    b_label_pred = np.argmax(logits, axis=1).tolist()
    label_pred.extend(b_label_pred)

print(confusion_matrix(labels, label_pred), '\n')
print(classification_report(labels, label_pred, output_dict=True))
print(f1_score(labels, label_pred, average='micro'))
