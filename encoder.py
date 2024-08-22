import torch
import os
from cocoa_trainer import CocoaTrainer
from config import SEQUENCE_LENGTH, ENCODER_TYPE, TAU, LAM, ENCODER_EPOCHS, ENCODER_BATCH_SIZE, BALANCE_ENCODER_DATA, SAVE_MODEL_PATH
from utils import fetch_data

trainer = CocoaTrainer(ENCODER_TYPE, SEQUENCE_LENGTH, TAU, LAM)

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

if BALANCE_ENCODER_DATA:
    trainer.balance_data()
trainer.train(ENCODER_EPOCHS, ENCODER_BATCH_SIZE)
trainer.save_model(SAVE_MODEL_PATH)