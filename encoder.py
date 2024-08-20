import torch
import os
from cocoa_trainer import CocoaTrainer
from config import SEQUENCE_LENGTH, IS_TRANSFORMER_BASED
from utils import fetch_data

TAU = 5
LAM = 2
EPOCHS = 30
BATCH_SIZE = 50
BALANCE_DATA = False
MODEL_PATH = "./models/cocoa_encoder"

trainer = CocoaTrainer(IS_TRANSFORMER_BASED, SEQUENCE_LENGTH, TAU, LAM)

# Get data
BASE_DIRECTORY = "aligned_data"
ALL_DATES = os.listdir(BASE_DIRECTORY)
aligned_data = fetch_data(BASE_DIRECTORY, ALL_DATES)

patients = set()
removed_patients = ["040520231330", "020920240900", "030820241000"]

DATE_IDX = 12
PATIENT_NAME_IDX = 17
for key in aligned_data.keys():
    if key[:DATE_IDX] not in removed_patients:
        patients.add(key[:PATIENT_NAME_IDX])

for patient in patients:
    trainer.add_data(aligned_data[f"{patient}_pose_arr"], aligned_data[f"{patient}_move_arr"], 
                     aligned_data[f"{patient}_label_arr"])

if BALANCE_DATA:
    trainer.balance_data()
trainer.train(EPOCHS, BATCH_SIZE)
trainer.save_model(MODEL_PATH)