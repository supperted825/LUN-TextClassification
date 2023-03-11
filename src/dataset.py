import torch
from torch.utils import Dataset
from transformers import AutoTokenizer

import nlpaug.augmenter.word as naw
import nlpaug.flow as naf


class TextDataset(Dataset):
    
    def __init__(self, df, tokenizer_str, augment=True, tfidf=None):
        
        self.augment = augment
        self.tfidf = tfidf
        
        self.text = df['text'].tolist()
        self.cls = (df['cls'] - 1).tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
        
        self.augments = naf.Sequential([
            naw.RandomWordAug(action='swap'),
            naw.RandomWordAug(action='delete'),
            naw.RandomWordAug(action='crop'),
            naw.SynonymAug()
        ])
    
    def __len__(self):
        return len(self.cls)
    
    def __getitem__(self, idx):
        
        text, label = self.text[idx], self.cls[idx]
        
        if self.augment:
            text = self.augments(text)
        
        encoding_dict = self.tokenize(text)
        token_ids = encoding_dict['input_ids']
        att_masks = encoding_dict['attention_mask']
        
        return token_ids, att_masks, label
    
    def tokenize(self, text):
        return self.tokenizer.encode_plus(
            text,
            max_length=512,
            add_special_tokens = True,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )