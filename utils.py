import numpy as np

def match_length(label_arr, pose_arr, move_arr):
    minimum = min(len(label_arr), len(pose_arr), len(move_arr))
    return label_arr[:minimum], pose_arr[:minimum], move_arr[:minimum]

def change_labels_to_bool(label_arr):
    return np.array([[True] if "Stable" in x else [False] for x in label_arr])

def find_negatives(label_batch):
    THRESHOLD = 1
    negatives = []
    for idx, label_seq in enumerate(label_batch):
        i = 0
        for label in label_seq:
            if not label:
                i += 1
        if i > THRESHOLD:
            negatives.append(idx)
    return negatives