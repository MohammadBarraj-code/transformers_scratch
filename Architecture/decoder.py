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

        def self_attention_sublayer(x):
            output, _ = attention.multi_head_attention(x, x, x, src_mask)
            return output

        x = self.residual1(x, self_attention_sublayer)

        def cross_attention_sublayer(x):
            output, _ = self.cross_attention(x, encoder_output, encoder_output, target_mask)

            return output

        x = self.residual2(x, cross_attention_sublayer)

        x = self.residual3(x, self.ff)


        return x


class Decoder(nn.Module):
    def __init__(self, nb_layers, d_model, nb_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, nb_heads, d_ff, dropout) for _ in range(nb_layers)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask = None, target_mask = None):
        for layer in self.laeyers:
            x = layer(x, encoder_output, src_mask, target_mask)

        x = self.norm(x)

        return x

        