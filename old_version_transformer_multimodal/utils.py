import numpy as np
import random
import os
from sequence_dataset import SequenceDataset
from concatenate_data import process_data_for_patient
from sklearn.model_selection import train_test_split
from config import IS_RANDOM, RANDOM_SEED, TEST_SIZE, DEFLATION_FACTOR, THRESHOLD

def fetch_data(base_directory, all_dates):
    aligned_data = {}
    for date in all_dates:
        date_dir = os.path.join(base_directory, date)

        patients = []
        if os.path.isdir(date_dir) and not date.endswith('.DS_Store'):
            patients = os.listdir(date_dir)

        for patient in patients:
            patient_dir = os.path.join(date_dir, patient)
            if os.path.isdir(patient_dir) and not patient.endswith('.DS_Store'):
                label_data, pose_data, movement_data = process_data_for_patient(patient_dir)

                date_patient_key_label = f"{date}_{patient}_label_arr"
                date_patient_key_pose = f"{date}_{patient}_pose_arr"
                date_patient_key_move = f"{date}_{patient}_move_arr"

                aligned_data[date_patient_key_label] = label_data
                aligned_data[date_patient_key_pose] = pose_data
                aligned_data[date_patient_key_move] = movement_data
    return aligned_data

def balance_data(poseSeqs, moveSeqs, labelSeqs):
    if not IS_RANDOM:
        random.seed(RANDOM_SEED)
    neg_idxs = find_negatives(labelSeqs)
    pos_idxs = [i for i in range(len(labelSeqs)) if i not in neg_idxs]
    random.shuffle(neg_idxs)
    random.shuffle(pos_idxs)
    minimum = min(len(neg_idxs), len(pos_idxs))
    balancedIdxs = [arr[x] for x in range(minimum) for arr in (pos_idxs, neg_idxs)]
    return [poseSeqs[i] for i in balancedIdxs], [moveSeqs[i] for i in balancedIdxs], [labelSeqs[i] for i in balancedIdxs]
    
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
        if i >= THRESHOLD:
            negatives.append(idx)
    return negatives

def find_positives(label_batch):
    neg_idxs = find_negatives(label_batch)
    non_neg_sequences = [seq for i, seq in enumerate(label_batch) if i not in neg_idxs]
    return non_neg_sequences

def get_seq_label(label_seq):
    unstable_count = 0
    for label in label_seq:
        if not label:
            unstable_count += 1
    return [unstable_count < THRESHOLD]