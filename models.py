import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from loss import FocalLoss


class TransformerClassifier(nn.Module):
  
    def __init__(self, transformer='xlnet-base-cased', reinit_layers=0, focal_alpha=None):
        super(TransformerClassifier, self).__init__()
        self.reinit_layers = reinit_layers
        self.backbone = AutoModel.from_pretrained(transformer)
        self.pooler = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = torch.nn.Linear(768, 4)
    
        if focal_alpha is not None:
            self.loss = FocalLoss(alpha=focal_alpha)
        else:
            self.loss = nn.CrossEntropyLoss()
        
        for n in range(self.reinit_layers):
            self.backbone.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)
    
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = self.pooler(output.last_hidden_state[:, 0])
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return self.loss(logits, labels.long()) if labels is not None else logits