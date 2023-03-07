import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from loss import FocalLoss


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
        self.dropout = nn.Dropout(p=0.5)
    
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

        output = output.last_hidden_state[:, 0]
        
        x = self.fc1(output)
        x = F.relu(x)
        x = self.dropout(x)
        
        if tfidf_features is not None:
            x = torch.cat([x, tfidf_features], dim=1).float()
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return self.loss(x, labels.long()) if labels is not None else x