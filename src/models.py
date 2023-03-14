import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .loss import FocalLoss


def mean_pooling(model_output, attention_mask):
    
    token_embeddings = model_output[0]
    attention_mask = attention_mask.unsqueeze(-1)
    mask_expanded  = attention_mask.expand(token_embeddings.size()).float()
    
    embed_num = torch.sum(token_embeddings * mask_expanded, dim=1) 
    embed_den = torch.clamp(mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = embed_num / embed_den
    
    return F.normalize(sentence_embeddings, p=2, dim=1)


class TransformerClassifier(nn.Module):
  
    def __init__(self, transformer, tfidf_dim=0, reinit_layers=0, focal_alpha=None):

        super(TransformerClassifier, self).__init__()
        self.reinit_layers = reinit_layers
        self.tfidf_dim = tfidf_dim
        self.fc_dim = AutoConfig.from_pretrained(transformer).hidden_size
        self.backbone = AutoModel.from_pretrained(transformer)
        
        self.fc1 = nn.Linear(self.fc_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim + tfidf_dim, self.fc_dim + tfidf_dim)
        self.fc3 = nn.Linear(self.fc_dim + tfidf_dim, 4)
    
        if focal_alpha is not None:
            self.loss = FocalLoss(alpha=focal_alpha)
        else:
            self.loss = nn.CrossEntropyLoss()
        
        for n in range(self.reinit_layers):
            self.backbone.layer[-(n+1)].apply(self.backbone._init_weights)

    def forward(self, input_ids, tfidf_features=None, token_type_ids=None, attention_mask=None, labels=None):

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        output = mean_pooling(output, attention_mask)
        
        x = self.fc1(output)
        x = F.relu(x)
        x = F.dropout(x)
        
        if tfidf_features is not None:
            x = torch.cat([x, tfidf_features], dim=1).float()
        
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        
        x = self.fc3(x)
        
        return self.loss(x, labels.long()) if labels is not None else x
    
    
class MLPClassifier(nn.Module):
    
    def __init__(self, tfidf_dim, fc_dim=0):
        
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(tfidf_dim, fc_dim + tfidf_dim)
        self.fc2 = nn.Linear(fc_dim + tfidf_dim, 4)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, labels=None):
        
        x = x.float()
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)

        return self.loss(x, labels.long()) if labels is not None else x