import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, numpyArr, sequence_length=5):
        self.sequence_length = sequence_length
        self.X = torch.from_numpy(numpyArr).float()
        self.size = len(numpyArr)-sequence_length

    def __len__(self):
        return self.size

    def __getitem__(self, i): 
        if i <= self.size:
            x = self.X[i:i+self.sequence_length, :]
        else:
            x = torch.Tensor(0,self.X.shape[1])
        return x