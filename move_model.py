import torch
from sklearn.model_selection import train_test_split
from autoencoder import Autoencoder
from torch import optim, nn
from torch.utils.data import DataLoader
import numpy as np
from sequence_dataset import SequenceDataset

# Device agnostic
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_model(model, train_loader, test_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss().to(device)
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
def filter(labelArr, moveArr):
    filteredArr = []
    for i in range(min(len(labelArr), len(moveArr))):
        if "stable" in labelArr[i][1].lower():
            filteredArr.append(moveArr[i])
    return np.asarray(filteredArr)

def expand_dataset(numpyArr, expansion_factor):
    return [val for val in numpyArr for _ in range(expansion_factor)]

# Get data
label_arr = np.loadtxt("aligned_data/041720231030/P002/Labels/Front_full_labels.csv",
                 delimiter=",", dtype=str, skiprows=1)
move_arr = np.loadtxt("aligned_data/041720231030/P002/April_17_Run_1.csv",
                 delimiter=",", usecols=range(1,7), dtype=float, skiprows=1)

# Modify dataset
EXPANSION_FACTOR = 18
move_arr = expand_dataset(move_arr, EXPANSION_FACTOR)

filtered_move_arr = filter(label_arr, move_arr)

RANDOM_SEED = 42
SEQUENCE_LENGTH = 45
BATCH_SIZE = 64
N_FEATURES = 6

full_dataset = SequenceDataset( 
    filtered_move_arr,
    sequence_length=SEQUENCE_LENGTH
)

# Create train-test split
train_dataset, test_dataset = \
    train_test_split(full_dataset, test_size=0.15, random_state=RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train
model = Autoencoder(SEQUENCE_LENGTH, N_FEATURES, embedding_dim=72)
model, history = train_model(model, train_loader, test_loader, epochs=20)

# Save model
PATH = "./models/move_autoencoder"
torch.save(model, PATH)