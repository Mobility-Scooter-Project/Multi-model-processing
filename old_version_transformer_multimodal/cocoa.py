import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, nhead=8, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        # Linear trans to embedd
        self.embedding = nn.Linear(n_features, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embedding_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=2 * embedding_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False # Determines if batched tensors are nested
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        x = self.embedding(x)
        x = x + self.positional_encoding

        # rearrange
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        # print(x.shape)
        x = x[-1, :, :].unsqueeze(0)
        # print(x.shape)
        x = x.squeeze(0).squeeze(0)
        # print(x.shape)
        return x


class ConcatLinear(nn.Module):
    def __init__(self, x_n_features, y_n_features):
        super(ConcatLinear, self).__init__()
        self.x_in_features, self.y_in_features = x_n_features, y_n_features
        concat_features = x_n_features + y_n_features
        self.linear = nn.Linear(in_features=concat_features, out_features=concat_features).to(device)

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        xy = self.linear(xy)
        xy = torch.squeeze(xy, 0)
        _x = xy[:self.x_in_features]
        _y = xy[self.x_in_features:]
        return _x, _y


class Cocoa(nn.Module):
    def __init__(self, seq_len, x_n_features, y_n_features, embedding_dim=64, nhead=8, num_layers=2):
        super(Cocoa, self).__init__()
        self.x_encoder = TransformerEncoder(seq_len, x_n_features, embedding_dim, nhead, num_layers).to(device)
        self.y_encoder = TransformerEncoder(seq_len, y_n_features, embedding_dim, nhead, num_layers).to(device)
        self.x_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim).to(device)
        self.y_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim).to(device)
        self.concat_linear = ConcatLinear(x_n_features=embedding_dim, y_n_features=embedding_dim).to(device)

    def forward(self, x, y):
        x = self.x_encoder(x)
        x = self.x_linear(x)
        y = self.y_encoder(y)
        y = self.y_linear(y)
        x, y = self.concat_linear(x, y)
        return x, y
