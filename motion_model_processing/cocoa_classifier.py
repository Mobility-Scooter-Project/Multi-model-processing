import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CombinedModel(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16, num_heads=4, num_layers=2):
        super(CombinedModel, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        # Embedding layer and Positional encoding
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embed the input sequence
        x = self.embedding(x) + self.positional_encoding
        x = x.permute(1, 0, 2)

        #Transformer encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)

        x = self.linear(x)
        x = self.sigmoid(x)
        return x
