import os
import sys
import pandas as pd
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from transformers import AutoModel

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Current Device : {device}")

from transformers import BertModel
bert_embedding_model = BertModel.from_pretrained('monologg/kobert')
model_name = 'monologg/kobert'
batch_size = 16
lr = 3e-5
epochs = 3
hidden_size = 768

class CustomClassifier(nn.Module):

    def __init__(self, hidden_size: int, n_label: int, model_name: str):
      super(CustomClassifier, self).__init__()
    
      self.model = AutoModel.from_pretrained(model_name)
    
      dropout_rate = 0.1
      linear_layer_hidden_size = batch_size

    
      self.classifier = nn.Sequential(
                                      nn.Linear(hidden_size, linear_layer_hidden_size), 
                                      nn.ReLU(),
                                      # nn.BatchNorm1d(linear_layer_hidden_size),  # Relu -> BN -> Dropout
                                      nn.Dropout(dropout_rate),
                                      nn.Linear(linear_layer_hidden_size, n_label)
                                      #nn.Sigmoid()
                                      )


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
    
        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            )
        
        cls_token_last_hidden_states =  outputs['pooler_output']
        
        logits = self.classifier(cls_token_last_hidden_states)
        
        return logits


class CrossEntropyLoss:
    def __call__(self):
        return CrossEntropyLoss()


class CustomRegressor(nn.Module):

    def __init__(self, hidden_size=768):
        super(CustomRegressor, self).__init__()

        self.model = bert_embedding_model

        dropout_rate = 0.1
        linear_layer_hidden_size = 32

    
        self.classifier = nn.Sequential(
                                      nn.Linear(hidden_size, linear_layer_hidden_size), 
                                      nn.ReLU(),
                                      nn.Dropout(dropout_rate),
                                      nn.Linear(linear_layer_hidden_size, 1)
                                      )


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
    
        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            )
        
        cls_token_last_hidden_states =  outputs['pooler_output']
        
        logits = self.classifier(cls_token_last_hidden_states)
        
        return logits