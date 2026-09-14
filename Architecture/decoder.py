import attention, feed_forward_nn
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, nb_heads, dropout=0.1):
        super().__init__()
        self.self_attention = attention.multi_head_attention(d_model, nb_heads)

        self.cross_attention = attention.multi_head_attention(d_model, nb_heads)
        self.ff = feed_forward_nn.FeedForward(d_model, d_ff, dropout)
        self.residual1 = feed_forward_nn.ResidualConnections(d_model, dropout)
        self.residual2 = feed_forward_nn.ResidualConnections(d_model, dropout)
        self.residual3 = feed_forward_nn.ResidualConnections(d_model, dropout)


    def forward(self, x, encoder_output, src_mask = None, target_mask = None):

        x = self.residual1(x, lambda x: self.self_attention(x, x, x, target_mask)[0])

        cross_attention_weights = None

        def cross_attention(x):
            output, weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
            nonlocal cross_attention_weights
            cross_attention_weights = weights
            return output

        

        x = self.residual2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, None)[0])
        x = self.residual3(x, self.ff)

        


        return x, cross_attention_weights


class Decoder(nn.Module):
    def __init__(self, nb_layers, d_model, nb_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff,nb_heads, dropout) for _ in range(nb_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask = None, target_mask = None):
        cross_attention_weights= None
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)

        x = self.norm(x)

        return x, cross_attention_weights

        