import torch
import os
from cocoa_trainer import CocoaTrainer
from config import SEQUENCE_LENGTH, RANDOM_SEED, IS_RANDOM
from utils import fetch_data

if not IS_RANDOM:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

TAU = 5
LAM = 2
EPOCHS = 3
BATCH_SIZE = 50
BALANCE_DATA = True
MODEL_PATH = "./old_version_transformer_multimodal/models/cocoa_encoder"

trainer = CocoaTrainer(SEQUENCE_LENGTH, TAU, LAM, IS_RANDOM)

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

for patient in sorted(patients):
    trainer.add_data(aligned_data[f"{patient}_pose_arr"], aligned_data[f"{patient}_move_arr"], 
                     aligned_data[f"{patient}_label_arr"])

if BALANCE_DATA:
    trainer.balance_data()
trainer.train(EPOCHS, BATCH_SIZE)
trainer.save_model(MODEL_PATH)