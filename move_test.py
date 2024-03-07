from sequence_dataset import SequenceDataset
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

PATH = "./models/move_autoencoder_65"
CSV_PATH = "./testing/move_test_unstable.csv"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load(PATH, map_location=device)
SEQUENCE_LENGTH = 65
BATCH_SIZE = 14

move_arr = np.loadtxt(CSV_PATH, delimiter=",", usecols=range(1,7), dtype=float, skiprows=1)

def expand_dataset(numpyArr, expansion_factor):
    return np.asarray([val for val in numpyArr for _ in range(expansion_factor)])

# Modify dataset
EXPANSION_FACTOR = 18
move_arr = expand_dataset(move_arr, EXPANSION_FACTOR)

test_dataset = SequenceDataset( 
    move_arr,
    sequence_length=SEQUENCE_LENGTH
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test:
loss_fn = nn.MSELoss().to(device)
with torch.no_grad():
    for batch in iter(test_loader):
        for seq_true in batch:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = loss_fn(seq_pred, seq_true)
            print(f'Loss value: {loss.item()}')