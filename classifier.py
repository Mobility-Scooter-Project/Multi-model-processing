from sklearn.model_selection import train_test_split
import torch
import os
from utils import fetch_data
from config import RANDOM_SEED, POSE_N_FEATURES, MOVE_N_FEATURES, TEST_SIZE
from cocoa_classifier_trainer import CocoaClassifierTrainer

SEQUENCE_LENGTH = 6
BATCH_SIZE = 50
EPOCHS = 20
EMBEDDING_DIM = 16
MODEL_PATH = "./models/cocoa_encoder"
FREEZE_STATE = False
BALANCE_DATA = True

trainer = CocoaClassifierTrainer(SEQUENCE_LENGTH, BATCH_SIZE, POSE_N_FEATURES, MOVE_N_FEATURES, EMBEDDING_DIM)

# Get data
BASE_DIRECTORY = "aligned_data"
ALL_DATES = os.listdir(BASE_DIRECTORY)
aligned_data = fetch_data(BASE_DIRECTORY, ALL_DATES)

patients = set()
removed_patients = ["040520231330","030820241000","030820241100"]

DATE_IDX = 12
PATIENT_NAME_IDX = 17
for key in aligned_data.keys():
    if key[:DATE_IDX] not in removed_patients:
        patients.add(key[:PATIENT_NAME_IDX])
print(patients)

for patient in patients:
    trainer.add_data(aligned_data[f"{patient}_pose_arr"], aligned_data[f"{patient}_move_arr"], 
                     aligned_data[f"{patient}_label_arr"])

trainer.load_encoder(MODEL_PATH)
trainer.freeze_encoder(FREEZE_STATE)
if BALANCE_DATA:
    trainer.balance_data()
trainer.train(EPOCHS, BATCH_SIZE)
