model_name = "esm2_t33_650M_UR50D" # @param ["esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D", "esm2_t12_35M_UR50D", "esm2_t6_8M_UR50D"]
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import EsmForSequenceClassification, EsmForTokenClassification, AutoTokenizer

trainable_params = lambda x: sum(p.numel() for p in x.parameters() if p.requires_grad)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = EsmForTokenClassification.from_pretrained(f"facebook/{model_name}",
                                                  num_labels=1,
                                                  hidden_dropout_prob=0.15)

tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
model = model.to(DEVICE)

print(f"Model {model_name} loaded on {DEVICE}. Trainable parameters: {trainable_params(model)}")