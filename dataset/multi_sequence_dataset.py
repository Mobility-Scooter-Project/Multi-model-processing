import torch
from torch.utils.data import Dataset

class MultiSequenceDataset(Dataset):
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.datasets = []
        self.datasizes = []
        self.size = 0

    def add(self, numpyArr):
        self.datasets.append(numpyArr)
        self.size += len(numpyArr)-(self.sequence_length-1)
        self.datasizes.append(self.size)
    
    def __find_index(self, index):
        # Can be changed to binary search for better performance
        offset = 0
        for i, j in enumerate(self.datasizes):
            if index < j:
                dataset_index = i
                tensor_index = index - offset
                return dataset_index, tensor_index
            offset += j
        return None

    def __len__(self):
        return self.size

    def __getitem__(self, i): 
        if i <= self.size:
            dataset_index, index = self.__find_index(i)
            x = self.datasets[dataset_index][index:index+self.sequence_length, :]
        else:
            x = torch.Tensor(0,self.datasets[0].shape[1])
        return x