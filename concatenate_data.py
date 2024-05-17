import os
import pandas as pd
import numpy as np
from config import DEFLATION_FACTOR

def match_length(label_arr, pose_arr, move_arr):
    minimum = min(len(label_arr), len(pose_arr), len(move_arr))
    return label_arr[:minimum], pose_arr[:minimum], move_arr[:minimum]

def change_labels_to_bool(label_arr):
    return np.array([[True] if "Stable" in x else [False] for x in label_arr])

def concatenate_csv_files(directory_path):
    csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
    data_frames = []
    for file in csv_files:
        df = pd.read_csv(file, skiprows=1)
        data_frames.append(df)
    concatenated_df = pd.concat(data_frames, ignore_index=True)
    return concatenated_df

def concatenate_yolov7_data(directory_path):
    csv_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]
    data_frames = [pd.read_csv(file) for file in csv_files]
    concatenated_df = pd.concat(data_frames, ignore_index=True)
    return concatenated_df

def process_data_for_patient(base_dir):
    # Process label
    labels_dir = os.path.join(base_dir, "Labels")
    label_arr = None
    if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
        label_df = concatenate_csv_files(labels_dir)
        label_arr = label_df.to_numpy()
        label_arr = label_arr[::DEFLATION_FACTOR]
        label_arr = change_labels_to_bool(label_arr)

    # Process Yolov data
    yolov7_dir = os.path.join(base_dir, "Yolov7")
    pose_arr = None
    if os.path.exists(yolov7_dir and os.path.isdir(yolov7_dir)):
        yolov7_df = concatenate_yolov7_data(yolov7_dir)
        pose_arr = yolov7_df.to_numpy(dtype="float32")
        pose_arr = pose_arr[::DEFLATION_FACTOR]

    # process move and the rest
    move_arr = None
    if os.path.exists(base_dir) and os.path.isdir(base_dir):
        move_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f)) and f.endswith('.csv') and 'run' in f.lower()]
        if move_files:
            move_file_path = os.path.join(base_dir, move_files[0])
            move_arr = np.loadtxt(move_file_path, delimiter=",", usecols=range(1, 7), dtype="float32", skiprows=1)
            label_arr, pose_arr, move_arr = match_length(label_arr, pose_arr, move_arr)

    return label_arr, pose_arr, move_arr
