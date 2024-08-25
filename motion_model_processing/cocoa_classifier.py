import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CombinedModel(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(CombinedModel, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        # Encoder part
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encode
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        x = hidden_n.reshape((1, self.embedding_dim))

        x = self.linear(x)
        x = self.sigmoid(x)
        return x
