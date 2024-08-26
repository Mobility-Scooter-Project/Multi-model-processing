import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler
from cocoa_classifier import CombinedModel
from dataset import multi_sequence_dataset as data
from config import RANDOM_SEED, LEARNING_RATE, TEST_SIZE
from logger import Logger
from utils import balance_data, find_negatives, get_seq_label
import pandas as pd
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CocoaClassifierTrainer():
    def __init__(self, seq_len, batch_size, n_features, embedding_dim=16, logger=None):
        self.pose_data = data.MultiSequenceDataset(seq_len)
        self.label_data = data.MultiSequenceDataset(seq_len)
        self.model = CombinedModel(seq_len, n_features, embedding_dim)
        self.model.to(device)
        self.loss_fn = nn.BCELoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        self.batch_size = batch_size
        self.skip_batches = False
        self.logger = logger

    def add_data(self, pose_arr, label_arr):
        self.pose_data.add(pose_arr)
        self.label_data.add(label_arr)

    def balance_data(self):
        self.pose_data, self.label_data = balance_data(self.pose_data, self.label_data)

    def save_roc_data_to_csv(self, true_labels, pred_probs, model_name, subfolder="roc_data"):
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        file_path = os.path.join(subfolder, f'{model_name}_roc_data.csv')
        data = {'TrueLabels': true_labels, 'PredProbs': pred_probs}
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"ROC data saved to {file_path}")

    def train(self, epochs, batch_size):
        label_train_dataset, label_test_dataset = \
            train_test_split(self.label_data, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        pose_train_dataset, pose_test_dataset = \
            train_test_split(self.pose_data, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        G = torch.Generator()
        G.manual_seed(RANDOM_SEED)
        train_sampler = RandomSampler(data_source=label_train_dataset, generator=G)
        test_sampler = RandomSampler(data_source=label_test_dataset, generator=G)

        all_true_labels = []
        all_pred_probs = []

        for epoch in range(0, epochs):
            model = self.model.train()
            train_losses = []
            train_accs = []
            train_sampler_save = list(train_sampler)
            test_sampler_save = list(test_sampler)

            label_train_loader = DataLoader(label_train_dataset, batch_size=batch_size, sampler=train_sampler_save)
            pose_train_loader = DataLoader(pose_train_dataset, batch_size=batch_size, sampler=train_sampler_save)
            label_test_loader = DataLoader(label_test_dataset, batch_size=batch_size, sampler=test_sampler_save)
            pose_test_loader = DataLoader(pose_test_dataset, batch_size=batch_size, sampler=test_sampler_save)

            for label_batch, pose_batch in zip(iter(label_train_loader), iter(pose_train_loader)):

                for pose_true, label_true in zip(pose_batch, label_batch):
                    pose_true, label_true = pose_true.to(device), label_true.to(device)
                    pred = model(pose_true)
                    true = torch.tensor(get_seq_label(label_true), dtype=torch.float32).unsqueeze(1)
                    loss = self.loss_fn(pred, true)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_accs.append(torch.round(pred[0]) == true[0])
                    train_losses.append(loss.item())

            test_losses = []
            test_accs = []
            model = self.model.eval()
            with torch.no_grad():
                for label_test_batch, move_test_batch in zip(iter(label_test_loader),iter(pose_test_loader)):
                    # Skip batches without negative pairs
                    if not find_negatives(label_test_batch) and self.skip_batches:
                        continue
                    for pose_true, label_true in zip(move_test_batch, label_test_batch):
                        pose_true = pose_true.to(device)
                        pred = model(pose_true)
                        true = torch.tensor(get_seq_label(label_true), dtype=torch.float32).unsqueeze(1)
                        loss = self.loss_fn(pred, true)
                        test_accs.append(torch.round(pred[0]) == true[0])
                        test_losses.append(loss.item())

                        all_true_labels.append(true.item())
                        all_pred_probs.append(pred.item())

            train_loss = np.mean(train_losses)
            train_acc = np.mean(train_accs)
            test_loss = np.mean(test_losses)
            test_acc = np.mean(test_accs)
            print(f"Epoch {epoch}: train loss {train_loss} | train acc {train_acc}, test loss {test_loss} | test acc {test_acc}")

            self.logger.log_training_output(
                f"Epoch {epoch + 1}: train loss {train_loss:.4f} | train acc {train_acc:.4f} | test loss {test_loss:.4f} | test acc {test_acc:.4f}")
        self.logger.end_logging()
        self.save_roc_data_to_csv(all_true_labels, all_pred_probs, "transformer_encoder_pose_only", subfolder="roc_data")