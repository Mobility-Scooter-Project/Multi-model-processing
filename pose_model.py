import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np
from sequence_dataset import SequenceDataset

# Device agnostic
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_model(model, train_loader, test_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], test=[])

    for epoch in range(0, epochs):
        model = model.train()
        train_losses = []
        
        # Train data
        for batch in iter(train_loader):
            for seq_true in batch:
                optimizer.zero_grad()

                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = loss_fn(seq_pred, seq_true)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        
        test_losses = []
        model = model.eval()

        # Test data
        with torch.no_grad():
            for batch in iter(test_loader):
                for seq_true in batch:
                    seq_true = seq_true.to(device)
                    seq_pred = model(seq_true)

                    loss = loss_fn(seq_pred, seq_true)

                    test_losses.append(loss.item())
    
        # For graph
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        # history['train'].append(train_loss)
        # history['test'].append(test_loss)
    
        print(f"Epoch {epoch}: train loss {train_loss}, test loss {test_loss}")

    return model.eval(), history

# Filter: only keep stable poses
def filter(labelArr, poseArr):
    i=0
    filteredArr = poseArr.copy()
    for s in labelArr:
        if "sway" in s[1].lower():
            filteredArr = np.delete(filteredArr, i,0)
        else:
            i += 1
    return filteredArr

# Get data
label_arr = np.loadtxt("aligned_data/041720231030/P002/Labels/Front 4.mp4_labels.csv",
                 delimiter=",", dtype=str, skiprows=1)
pose_arr = np.loadtxt("aligned_data/041720231030/P002/Yolov7/Front_4.csv",
                 delimiter=",", dtype=float, skiprows=1)

filtered_pose_arr = filter(label_arr, pose_arr)

RANDOM_SEED = 42
SEQUENCE_LENGTH = 4
BATCH_SIZE = 64
N_FEATURES = 18

full_dataset = SequenceDataset( 
    filtered_pose_arr,
    sequence_length=SEQUENCE_LENGTH
)

# Create train-test split
train_dataset, test_dataset = \
    train_test_split(full_dataset, test_size=0.15, random_state=RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train
model = Autoencoder(SEQUENCE_LENGTH, N_FEATURES, embedding_dim=324)
model, history = train_model(model, train_loader, test_loader, epochs=150)

# Save model
PATH = "./models/lstm_autoencoder"
torch.save(model, PATH)