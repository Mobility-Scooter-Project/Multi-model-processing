import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from torch import optim, nn
import numpy as np

# Device agnostic
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_dataset(numpyArr):
  sequences = numpyArr.tolist()
  dataset = [torch.tensor(s).unsqueeze(0) for s in sequences]
  n_seq, seq_len, n_features = torch.stack(dataset).shape
  return dataset, seq_len, n_features

def train_model(model, train_dataset, test_dataset, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss(reduction='sum').to(device)

    history = dict(train=[], test=[])

    for epoch in range(0, epochs):
        model = model.train()
        train_losses = []
        
        # Train data
        for seq_true in train_dataset:
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
            for seq_true in test_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = loss_fn(seq_pred, seq_true)

                test_losses.append(loss.item())
    
        # For graph
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        history['train'].append(train_loss)
        history['test'].append(test_loss)
    
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

# Create train-test split
RANDOM_SEED = 42

train_arr, test_arr = \
    train_test_split(filtered_pose_arr, test_size=0.15, random_state=RANDOM_SEED)

train_dataset, seq_len, n_features = create_dataset(train_arr)
test_dataset, __, __ = create_dataset(test_arr)


model = Autoencoder(seq_len, n_features, embedding_dim=324)
model, history = train_model(model, train_dataset, test_dataset, epochs=150)

# Save model
PATH = "./models/lstm_autoencoder"
torch.save(model, PATH)


        


