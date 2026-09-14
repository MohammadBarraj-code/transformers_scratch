import torch.nn as nn
import positional_encoding, embeddings, encoder, decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab, target_vocab, d_model=512, nb_heads=8, nb_layers=6,
                 d_ff=2048, max_seq_length=512, dropout=0.1):
        super().__init__()

        self.source_embedding = embeddings.Embedder(src_vocab, d_model)
        self.target_embedding = embeddings.Embedder(target_vocab, d_model)

        self.source_pe = positional_encoding.PositionalEncoding(d_model, max_seq_length, dropout)
        self.target_pe = positional_encoding.PositionalEncoding(d_model, max_seq_length,dropout)

        self.encoder = encoder.Encoder(nb_layers, d_model, nb_heads, d_ff, dropout)
        self.decoder = decoder.Decoder(nb_layers, d_model, nb_heads, d_ff, dropout)

        self.linear = nn.Linear(d_model, target_vocab)


    def forward(self, source, target, src_mask=None, target_mask=None):
        source = self.source_embedding(source)
        source = self.source_pe(source)
        source = self.encoder(source, src_mask)

        target = self.target_embedding(target)
        target = self.target_pe(target)

        target, attention_weights = self.decoder(target, source,src_mask, target_mask)

        target = self.linear(target)

        return target, attention_weights
