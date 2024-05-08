from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler
from sequence_dataset import SequenceDataset
from cocoa_loss import CocoaLoss
from cocoa import Cocoa
from utils import match_length, change_labels_to_bool, find_negatives 
from config import RANDOM_SEED, POSE_N_FEATURES, MOVE_N_FEATURES, TEST_SIZE

# Device agnostic
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Get data
# Current implementation deflates the number of pose to match the number of move points
# Another implementation may be to inflate the number of move sequences to match number of pose sequences
DEFLATION_FACTOR = 18 # Video frames to movement frames ratio is 18:1 
label_arr = np.loadtxt("aligned_data/041720231030/P002/Labels/Front_full_labels.csv",
                 delimiter=",", dtype=str, skiprows=1)
label_arr = label_arr[::DEFLATION_FACTOR]
label_arr = change_labels_to_bool(label_arr)

pose_arr = np.loadtxt("aligned_data/041720231030/P002/Yolov7/Front_full.csv",
                 delimiter=",", dtype="float32", skiprows=1)
pose_arr = pose_arr[::DEFLATION_FACTOR]

move_arr = np.loadtxt("aligned_data/041720231030/P002/April_17_Run_1.csv",
                 delimiter=",", usecols=range(1,7), dtype="float32", skiprows=1)

label_arr, pose_arr, move_arr = match_length(label_arr, pose_arr, move_arr)

SEQUENCE_LENGTH = 6
BATCH_SIZE = 50
TAU = 5
LAM = 2
EPOCHS = 20

label_dataset = SequenceDataset( 
    label_arr,
    sequence_length=SEQUENCE_LENGTH
)

pose_dataset = SequenceDataset( 
    pose_arr,
    sequence_length=SEQUENCE_LENGTH
)

move_dataset = SequenceDataset( 
    move_arr,
    sequence_length=SEQUENCE_LENGTH
)

# Create train-test split
label_train_dataset, label_test_dataset = \
    train_test_split(label_dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)

pose_train_dataset, pose_test_dataset = \
    train_test_split(pose_dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)

move_train_dataset, move_test_dataset = \
    train_test_split(move_dataset, test_size=TEST_SIZE, random_state=RANDOM_SEED)

def train_model(model, tau, lam, epochs, label_train_dataset, label_test_dataset, pose_train_dataset, pose_test_dataset, 
                move_train_dataset, move_test_dataset):
    G = torch.Generator()
    G.manual_seed(RANDOM_SEED)
    train_sampler = RandomSampler(data_source=label_train_dataset, generator=G)
    test_sampler = RandomSampler(data_source=label_test_dataset, generator=G)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = CocoaLoss(tau, lam).to(device)

    for epoch in range(0, epochs):
        model = model.train()
        train_losses = []
        train_sampler_save = list(train_sampler)
        test_sampler_save = list(test_sampler)

        label_train_loader = DataLoader(label_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler_save)
        pose_train_loader = DataLoader(pose_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler_save)
        move_train_loader = DataLoader(move_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler_save)
        label_test_loader = DataLoader(label_test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler_save)
        pose_test_loader = DataLoader(pose_test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler_save)
        move_test_loader = DataLoader(move_test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler_save)


        for label_batch, pose_batch, move_batch in zip(iter(label_train_loader), iter(pose_train_loader), iter(move_train_loader)):
            optimizer.zero_grad()
            # Skip batches without negative pairs
            if not find_negatives(label_batch):
                continue
            pred_pose_batch = []
            pred_move_batch = []
            for pose_true, move_true in zip(pose_batch, move_batch):
                pose_true, move_true = pose_true.to(device), move_true.to(device)
                pose_pred, move_pred = model(pose_true, move_true)
                pred_pose_batch.append(pose_pred)
                pred_move_batch.append(move_pred)
            loss = loss_fn(pred_pose_batch, pred_move_batch, label_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_losses = []
        model = model.eval()

        with torch.no_grad():
            for label_test_batch, pose_test_batch, move_test_batch in zip(iter(label_test_loader), iter(pose_test_loader), iter(move_test_loader)):
                if not find_negatives(label_test_batch):
                    continue
                pred_pose_batch = []
                pred_move_batch = []
                for pose_true, move_true in zip(pose_test_batch, move_test_batch):
                    pose_true, move_true = pose_true.to(device), move_true.to(device)
                    pose_pred, move_pred = model(pose_true, move_true)
                    pred_pose_batch.append(pose_pred)
                    pred_move_batch.append(move_pred)
                loss = loss_fn(pred_pose_batch, pred_move_batch, label_test_batch)
                test_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        print(f"Epoch {epoch}: train loss {train_loss}, test loss {test_loss}")

# Creating model:

model = Cocoa(SEQUENCE_LENGTH, POSE_N_FEATURES, MOVE_N_FEATURES, embedding_dim=16)
model.to(device)

train_model(model, TAU, LAM, EPOCHS, label_train_dataset, label_test_dataset, pose_train_dataset, pose_test_dataset, 
                move_train_dataset, move_test_dataset)

# Save model
PATH = "./models/cocoa_encoder"
torch.save(model.state_dict(), PATH)
