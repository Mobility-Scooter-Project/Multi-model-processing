from sklearn.model_selection import train_test_split
import torch
import os
from utils import fetch_data
from config import POSE_N_FEATURES, MOVE_N_FEATURES, SEQUENCE_LENGTH, CLASSIFIER_BATCH_SIZE, EMBEDDING_DIM, CLASSIFIER_EPOCHS, BALANCE_CLASSIFIER_DATA, \
                   FREEZE_ENCODER_STATE, LOAD_MODEL_PATH, CLASSIFIER_ENCODER_TYPE, N_HEAD, N_LAYERS
from cocoa_classifier_trainer import CocoaClassifierTrainer
from logger import Logger

hyperparams = {
    'ENCODER_TYPE': CLASSIFIER_ENCODER_TYPE,
    'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
    'BATCH_SIZE': CLASSIFIER_BATCH_SIZE,
    'EPOCHS': CLASSIFIER_EPOCHS,
    'EMBEDDING_DIM': EMBEDDING_DIM,
    'FREEZE_STATE': FREEZE_ENCODER_STATE,
}
logger = Logger()
log_file_path = logger.start_logging(file_name_prefix="classifier")
logger.log_hyperparameters(hyperparams)

trainer = CocoaClassifierTrainer(SEQUENCE_LENGTH, CLASSIFIER_BATCH_SIZE, POSE_N_FEATURES, MOVE_N_FEATURES, CLASSIFIER_ENCODER_TYPE, EMBEDDING_DIM, N_HEAD, N_LAYERS, logger)

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

trainer.load_encoder(LOAD_MODEL_PATH)
trainer.freeze_encoder(FREEZE_ENCODER_STATE)
if BALANCE_CLASSIFIER_DATA:
    trainer.balance_data()
trainer.train(CLASSIFIER_EPOCHS, CLASSIFIER_BATCH_SIZE)
