import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
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
    
class CocoaLinearTransformer(nn.Module):
  def __init__(self, seq_len, x_n_features, y_n_features, embedding_dim=16, nhead=8, nlayers=6):
    super(CocoaLinearTransformer, self).__init__()
    self.x_encoder = nn.Linear(in_features=seq_len*x_n_features, out_features=embedding_dim).to(device)
    self.y_encoder = nn.Linear(in_features=seq_len*y_n_features, out_features=embedding_dim).to(device)
    x_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
    self.x_transformer = nn.TransformerEncoder(x_encoder_layer, nlayers)
    y_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
    self.y_transformer = nn.TransformerEncoder(y_encoder_layer, nlayers)
    self.x_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim).to(device)
    self.y_linear = nn.Linear(in_features=embedding_dim, out_features=embedding_dim).to(device)    
    self.concat_linear = ConcatLinear(x_n_features=embedding_dim, y_n_features=embedding_dim).to(device)

  def forward(self, x, y):
    x = torch.unsqueeze(torch.flatten(x), 0)
    x = self.x_encoder(x)
    x = self.x_transformer(x)
    x = self.x_linear(x)
    y = torch.unsqueeze(torch.flatten(y), 0)
    y = self.y_encoder(y)
    y = self.y_transformer(y)
    y = self.y_linear(y)
    x, y = self.concat_linear(x,y)
    return x, y
  