import torch
from torch import nn
import numpy as np
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader

PATH = "./models/pose_autoencoder_65"
CSV_PATH = "./testing/pose_test_stable.csv"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load(PATH, map_location=device)
SEQUENCE_LENGTH = 65
BATCH_SIZE = 1

pose_arr = np.loadtxt(CSV_PATH, delimiter=",", dtype=float, skiprows=1)

test_dataset = SequenceDataset( 
    pose_arr,
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