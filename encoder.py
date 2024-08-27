import torch
import os
from cocoa_trainer import CocoaTrainer
from config import RANDOM_SEED, SEQUENCE_LENGTH, ENCODER_TYPE, TAU, LAM, ENCODER_EPOCHS, ENCODER_BATCH_SIZE, BALANCE_ENCODER_DATA, SAVE_MODEL_PATH, N_HEAD, N_LAYER, EMBEDDING_DIM
from utils import fetch_data
from logger import Logger

torch.use_deterministic_algorithms(True)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

hyperparams = {
    'ENCODER_TYPE': ENCODER_TYPE,
    'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
    'BATCH_SIZE': ENCODER_BATCH_SIZE,
    'EPOCHS': ENCODER_EPOCHS,
    'EMBEDDING_DIM': EMBEDDING_DIM,
    'TAU': TAU,
    'LAM': LAM,
    'N_HEAD': N_HEAD,
    'N_LAYER': N_LAYER
}
logger = Logger(log_dir = "encoder_logs")

log_file_path = logger.start_logging(file_name_prefix="encoder")
logger.log_hyperparameters(hyperparams)

trainer = CocoaTrainer(ENCODER_TYPE, SEQUENCE_LENGTH, TAU, LAM, N_HEAD, N_LAYER, logger)

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

for patient in sorted(patients):
    trainer.add_data(aligned_data[f"{patient}_pose_arr"], aligned_data[f"{patient}_move_arr"], 
                     aligned_data[f"{patient}_label_arr"])

if BALANCE_ENCODER_DATA:
    trainer.balance_data()
trainer.train(ENCODER_EPOCHS, ENCODER_BATCH_SIZE)
trainer.save_model(SAVE_MODEL_PATH)