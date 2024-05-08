import numpy as np
from sequence_dataset import SequenceDataset
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, TEST_SIZE, DEFLATION_FACTOR, THRESHOLD

def fetch_data(pose_path, move_path, label_path, sequence_len, test_size=TEST_SIZE, deflation_factor=DEFLATION_FACTOR):
    label_arr = np.loadtxt(label_path,
                    delimiter=",", dtype=str, skiprows=1)
    label_arr = label_arr[::deflation_factor]
    label_arr = change_labels_to_bool(label_arr)

    pose_arr = np.loadtxt(pose_path,
                    delimiter=",", dtype="float32", skiprows=1)
    pose_arr = pose_arr[::deflation_factor]

    move_arr = np.loadtxt(move_path,
                    delimiter=",", usecols=range(1,7), dtype="float32", skiprows=1)

    label_arr, pose_arr, move_arr = match_length(label_arr, pose_arr, move_arr)
    label_dataset = SequenceDataset( 
        label_arr,
        sequence_length=sequence_len
    )

    pose_dataset = SequenceDataset( 
        pose_arr,
        sequence_length=sequence_len
    )

    move_dataset = SequenceDataset( 
        move_arr,
        sequence_length=sequence_len
    )

    # Create train-test split
    label_train_dataset, label_test_dataset = \
        train_test_split(label_dataset, test_size=test_size, random_state=RANDOM_SEED)

    pose_train_dataset, pose_test_dataset = \
        train_test_split(pose_dataset, test_size=test_size, random_state=RANDOM_SEED)

    move_train_dataset, move_test_dataset = \
        train_test_split(move_dataset, test_size=test_size, random_state=RANDOM_SEED)
    return pose_train_dataset, pose_test_dataset, move_train_dataset, move_test_dataset, label_train_dataset, label_test_dataset

def match_length(label_arr, pose_arr, move_arr):
    minimum = min(len(label_arr), len(pose_arr), len(move_arr))
    return label_arr[:minimum], pose_arr[:minimum], move_arr[:minimum]

def change_labels_to_bool(label_arr):
    return np.array([[True] if "Stable" in x else [False] for x in label_arr])

def find_negatives(label_batch):
    negatives = []
    for idx, label_seq in enumerate(label_batch):
        i = 0
        for label in label_seq:
            if not label:
                i += 1
        if i > THRESHOLD:
            negatives.append(idx)
    return negatives

def get_seq_label(label_seq):
    unstable_count = 0
    for label in label_seq:
        if not label:
            unstable_count += 1
    return [unstable_count < THRESHOLD]