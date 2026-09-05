import attention, feed_forward_nn
import torch.nn as nn
class EncoderBlock(nn.Module):
    def __init__(self, d_model, nb_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = attention.multi_head_attention(d_model, nb_heads)
        self.ff = feed_forward_nn.FeedForward(d_model, d_ff, dropout)
        self.residual1 = feed_forward_nn.ResidualConnections(d_model, dropout)
        self.residual2 = feed_forward_nn.ResidualConnections(d_model, dropout)


    def forward(self, x, mask=None):

        def attention_sublayer(x):
            output, _ = self.attention(x, x, x, mask)
            return output

        x = self.residual1(x, attention_sublayer)
        x = self.residual2(x, self.ff)

        return x