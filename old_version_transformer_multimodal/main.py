import torch
import os
from cocoa_trainer import CocoaTrainer
from config import SEQUENCE_LENGTH, RANDOM_SEED, IS_RANDOM, EMBEDDING_DIM
from utils import fetch_data
from cocoa_classifier_trainer import CocoaClassifierTrainer
from logger import Logger

if not IS_RANDOM:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

TAU = 5
LAM = 2
EPOCHS = 3
BATCH_SIZE = 50
BALANCE_DATA = True
# Classifier train
FREEZE_STATE = False
LOAD_ENCODER = False

encoder_trainer = CocoaTrainer(SEQUENCE_LENGTH, TAU, LAM, IS_RANDOM)

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
    encoder_trainer.add_data(aligned_data[f"{patient}_pose_arr"], aligned_data[f"{patient}_move_arr"], 
                     aligned_data[f"{patient}_label_arr"])

if BALANCE_DATA:
    encoder_trainer.balance_data()
encoder_trainer.train(EPOCHS, BATCH_SIZE)

hyperparams = {
    'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
    'BATCH_SIZE': BATCH_SIZE,
    'EPOCHS': EPOCHS,
    'EMBEDDING_DIM': EMBEDDING_DIM,
    'FREEZE_STATE': FREEZE_STATE,
}

logger = Logger()
log_file_path = logger.start_logging(file_name_prefix="classifier")
logger.log_hyperparameters(hyperparams)

# Classifier train
classifier_trainer = CocoaClassifierTrainer(SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_DIM, IS_RANDOM, logger)

for patient in sorted(patients):
    classifier_trainer.add_data(aligned_data[f"{patient}_pose_arr"], aligned_data[f"{patient}_move_arr"], 
                     aligned_data[f"{patient}_label_arr"])

if LOAD_ENCODER:
    classifier_trainer.set_encoder(encoder_trainer.get_model())
classifier_trainer.freeze_encoder(FREEZE_STATE)
if BALANCE_DATA:
    classifier_trainer.balance_data()
classifier_trainer.train(EPOCHS, BATCH_SIZE)