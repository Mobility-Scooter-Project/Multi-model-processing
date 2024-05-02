import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim 
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

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((1, self.embedding_dim))
  
class ConcatLinear(nn.Module):
  def __init__(self, x_n_features, y_n_features):
    super(ConcatLinear, self).__init__()
    self.x_in_features, self.y_in_features = x_n_features, y_n_features
    concat_features = x_n_features+y_n_features
    self.linear = nn.Linear(in_features=concat_features, out_features=concat_features).to(device)
  
  def forward(self, x,y):
    xy = torch.cat([x,y], dim=-1)
    xy = self.linear(xy)
    xy = torch.squeeze(xy, 0)
    _x = xy[:self.x_in_features]
    _y = xy[self.x_in_features:]
    return _x, _y
    
class Cocoa(nn.Module):
  def __init__(self, seq_len, x_n_features, y_n_features, embedding_dim=64):
    super(Cocoa, self).__init__()
    self.x_encoder = Encoder(seq_len, x_n_features, embedding_dim).to(device)
    self.y_encoder = Encoder(seq_len, y_n_features, embedding_dim).to(device)
    self.x_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim).to(device)
    self.y_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim).to(device)
    self.concat_linear = ConcatLinear(x_n_features=embedding_dim, y_n_features=embedding_dim).to(device)

  def forward(self, x, y):
    x = self.x_encoder(x)
    x = self.x_linear(x)
    y = self.y_encoder(y)
    y = self.y_linear(y)
    x, y = self.concat_linear(x,y)
    return x, y
  