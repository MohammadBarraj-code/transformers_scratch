import torch, torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length = 12, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.positional_encoding = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)
        #Sinsoidal term: position/10,000*(2i/d_model)
        #Equal to: position * 10,000^-(2i/d_model)
        #a^b = e^(b*ln(a))
        sinusoidal_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000) / d_model))

        self.positional_encoding[:, 0::2] = torch.sin(position * sinusoidal_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * sinusoidal_term)

        self.positional_encoding = self.positional_encoding.unsqueeze(0)

        #Specifying for the model that this is part of the model but not a trainable parameter
        self.register_buffer('positional_encoding', self.positional_encoding)


    def forward(self, x):
        sequence_length = x.size(1)

        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x)

        return x