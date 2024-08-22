import torch
import torch.nn as nn
from config import encoder_type
from cocoa_lstm import CocoaLstm
from cocoa_lstm_transformer import CocoaLstmTransformer
from cocoa_linear_transformer import CocoaLinearTransformer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ConcatLinear(nn.Module):
  def __init__(self, x_n_features, y_n_features, out_features):
    super(ConcatLinear, self).__init__()
    self.x_in_features, self.y_in_features = x_n_features, y_n_features
    concat_features = x_n_features+y_n_features
    self.linear = nn.Linear(in_features=concat_features, out_features=out_features).to(device)

  def forward(self, x, y):
    xy = self.__concat_array(x,y)
    xy = self.linear(xy)
    xy = torch.squeeze(xy, 0)
    return xy
  
  def  __concat_array(self, x,y):
    xy = torch.cat([x,y], dim=-1)
    return xy

class CocoaClassifier(nn.Module):
  def __init__(self, seq_len, x_n_features, y_n_features, encoder_type, embedding_dim, nhead, nlayers):
    super(CocoaClassifier, self).__init__()
    match encoder_type:
      case encoder_type.LSTM:
          self.cocoa = CocoaLstm(seq_len, x_n_features, y_n_features, embedding_dim)
      case encoder_type.LSTM_TRANSFORMER:
          self.cocoa = CocoaLstmTransformer(seq_len, x_n_features, y_n_features, embedding_dim, nhead, nlayers)
      case encoder_type.LINEAR_TRANSFORMER:
          self.cocoa = CocoaLinearTransformer(seq_len, x_n_features, y_n_features, embedding_dim, nhead, nlayers)
      case _:
          raise Exception("Invalid encoder type")
    self.cocoa.to(device)
    concat_features = embedding_dim*2
    self.concatLinear = ConcatLinear(embedding_dim, embedding_dim, out_features=concat_features)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(in_features=concat_features, out_features=1)
    self.sigmoid = nn.Sigmoid()

  def load_encoder(self, path):
    # Save from GPU, load to GPU
    self.cocoa.load_state_dict(torch.load(path))
    self.cocoa.to(device)

  def freeze_encoder(self, bool):
    self.cocoa.requires_grad_(not bool)

  def forward(self, x, y):
    x, y = self.cocoa(x,y)
    # This implementation contains an extra linear layer w/ ReLu:
    xy = self.relu(self.concatLinear(x, y))
    xy = self.sigmoid(self.linear(xy))
    return xy