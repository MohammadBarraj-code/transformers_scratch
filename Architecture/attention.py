import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    numerator = Q @ K.transpose(-2, -1)
    denom = torch.sqrt(d_k)
    scores = numerator/denom

    if mask is not None:
        scores = scores.mask_fill(~mask, -1e9)

    attention_weights = torch.softmax(scores, dim=-1)

    output = attention_weights @ V


    return output, attention_weights


class multi_head_attention(nn.Module):
    def __init__(self, d_model, nb_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // nb_heads
        self.nb_heads = nb_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask = None):
        batch_size = Q.size[0]

        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)

        Q = Q.view(batch_size, -1, self.nb_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.nb_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.nb_heads, self.d_k).transpose(1, 2)

        output, attention_weights = scaled_dot_product(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.nb_heads*self.d_k)

        output = self.w_o(output)

        return output, attention_weights        

    