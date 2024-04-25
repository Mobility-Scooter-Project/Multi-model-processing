from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from sequence_dataset import SequenceDataset

# Device agnostic
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def match_length(label_arr, pose_arr, move_arr):
    minimum = min(len(label_arr), len(pose_arr), len(move_arr))
    return label_arr[:minimum], pose_arr[:minimum], move_arr[:minimum]

def change_labels_to_bool(label_arr):
    return np.array([[True] if "Stable" in x else [False] for x in label_arr])

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

RANDOM_SEED = 42
SEQUENCE_LENGTH = 2
BATCH_SIZE = 5
POSE_N_FEATURES = 18
MOVE_N_FEATURES = 6
TEST_SIZE = 0.15

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


def train_model(model, label_train_dataset, label_test_dataset, pose_train_dataset, pose_test_dataset, 
                move_train_dataset, move_test_dataset):
    G = torch.Generator()
    G.manual_seed(RANDOM_SEED)
    train_sampler = RandomSampler(data_source=label_train_dataset, generator=G)
    test_sampler = RandomSampler(data_source=label_test_dataset, generator=G)

    # for iter in epoch
    train_sampler_save = list(train_sampler)
    test_sampler_save = list(test_sampler)

    label_train_loader = DataLoader(label_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler_save)
    pose_train_loader = DataLoader(pose_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler_save)
    move_train_loader = DataLoader(move_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler_save)

    for label_batch, pose_batch, move_batch in zip(iter(label_train_loader), iter(pose_train_loader), iter(move_train_loader)):
        pred_pose_batch = []
        pred_move_batch = []
        for pose_true, move_true in zip(pose_batch, move_batch):
            pose_true, move_true = pose_true.to(device), move_true.to(device)
            pose_pred, move_pred = model(pose_true, move_true)
            pred_pose_batch.append(pose_pred)
            pred_move_batch.append(move_pred)
        # loss = loss_fn(pred_pose_batch, pred_move_batch, label_batch)

        #TODO: Create loss function and implement training
        # loss = loss / accum_iter 
        # optimizer.step()
        # optimizer.zero_grad()
        

# train_model(None, label_train_dataset, label_test_dataset, pose_train_dataset, pose_test_dataset, 
#                 move_train_dataset, move_test_dataset)

    
