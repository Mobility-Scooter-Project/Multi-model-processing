from sklearn.model_selection import train_test_split
import torch
from utils import fetch_data
from config import RANDOM_SEED, POSE_N_FEATURES, MOVE_N_FEATURES, TEST_SIZE
from cocoa_classifier_trainer import CocoaClassifierTrainer

# Device agnostic
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

POSE_PATH = "aligned_data/041720231030/P002/Yolov7/Front_full.csv"
MOVE_PATH = "aligned_data/041720231030/P002/April_17_Run_1.csv"
LABEL_PATH = "aligned_data/041720231030/P002/Labels/Front_full_labels.csv"
SEQUENCE_LENGTH = 6
BATCH_SIZE = 50
EPOCHS = 20
EMBEDDING_DIM = 16
MODEL_PATH = "PATH"
FREEZE_STATE = True


pose_train_dataset, pose_test_dataset, move_train_dataset, move_test_dataset, label_train_dataset, label_test_dataset = \
    fetch_data(POSE_PATH, MOVE_PATH, LABEL_PATH, SEQUENCE_LENGTH, TEST_SIZE)

trainer = CocoaClassifierTrainer(SEQUENCE_LENGTH, BATCH_SIZE, POSE_N_FEATURES, MOVE_N_FEATURES, EMBEDDING_DIM)
trainer.load_encoder(MODEL_PATH)
trainer.freeze_encoder(FREEZE_STATE)
trainer.train(EPOCHS, label_train_dataset, label_test_dataset, pose_train_dataset, 
              pose_test_dataset, move_train_dataset, move_test_dataset)
