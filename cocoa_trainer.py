import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from config import RANDOM_SEED, LEARNING_RATE, POSE_N_FEATURES, MOVE_N_FEATURES, EMBEDDING_DIM, TEST_SIZE, encoder_type
from utils import balance_data, find_negatives
from dataset import multi_sequence_dataset as data
from cocoa_loss import CocoaLoss
from cocoa_lstm import CocoaLstm
from cocoa_lstm_transformer import CocoaLstmTransformer
from cocoa_linear_transformer import CocoaLinearTransformer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CocoaTrainer():
    def __init__(self, encoder_type, seq_len, tau, lam, nhead, nlayer) -> None:
        self.pose_data = data.MultiSequenceDataset(seq_len)
        self.move_data = data.MultiSequenceDataset(seq_len)
        self.label_data = data.MultiSequenceDataset(seq_len)
        match encoder_type:
            case encoder_type.LSTM:
                self.model = CocoaLstm(seq_len, POSE_N_FEATURES, MOVE_N_FEATURES, EMBEDDING_DIM)
            case encoder_type.LSTM_TRANSFORMER:
                self.model = CocoaLstmTransformer(seq_len, POSE_N_FEATURES, MOVE_N_FEATURES, EMBEDDING_DIM, nhead, nlayer)
            case encoder_type.LINEAR_TRANSFORMER:
                self.model = CocoaLinearTransformer(seq_len, POSE_N_FEATURES, MOVE_N_FEATURES, EMBEDDING_DIM, nhead, nlayer)
            case _:
                raise Exception("Invalid encoder type")
        self.model.to(device)
        self.loss_fn = CocoaLoss(tau, lam).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.skip_batches = False

    def add_data(self, pose_arr, move_arr, label_arr):
        self.pose_data.add(pose_arr)
        self.move_data.add(move_arr)
        self.label_data.add(label_arr)

    def balance_data(self):
        self.pose_data, self.move_data, self.label_data = balance_data(self.pose_data, self.move_data, self.label_data)

    def get_model(self):
        return self.model
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def skip_nonneg_batches(self, bool):
        self.skip_batches = bool

    def train(self, epochs, batch_size):
        label_train_dataset, label_test_dataset = \
            train_test_split(self.label_data, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        pose_train_dataset, pose_test_dataset = \
            train_test_split(self.pose_data, test_size=TEST_SIZE, random_state=RANDOM_SEED)

        move_train_dataset, move_test_dataset = \
            train_test_split(self.move_data, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        G = torch.Generator()
        G.manual_seed(RANDOM_SEED)
        train_sampler = RandomSampler(data_source=label_train_dataset, generator=G)
        test_sampler = RandomSampler(data_source=label_test_dataset, generator=G)
        for epoch in range(0, epochs):
            self.model = self.model.train()
            train_losses = []
            train_sampler_save = list(train_sampler)
            test_sampler_save = list(test_sampler)
            label_train_loader = DataLoader(label_train_dataset, batch_size=batch_size, sampler=train_sampler_save)
            pose_train_loader = DataLoader(pose_train_dataset, batch_size=batch_size, sampler=train_sampler_save)
            move_train_loader = DataLoader(move_train_dataset, batch_size=batch_size, sampler=train_sampler_save)
            label_test_loader = DataLoader(label_test_dataset, batch_size=batch_size, sampler=test_sampler_save)
            pose_test_loader = DataLoader(pose_test_dataset, batch_size=batch_size, sampler=test_sampler_save)
            move_test_loader = DataLoader(move_test_dataset, batch_size=batch_size, sampler=test_sampler_save)

            for label_batch, pose_batch, move_batch in zip(iter(label_train_loader), iter(pose_train_loader), iter(move_train_loader)):
                self.optimizer.zero_grad()
                # Skip batches without negative pairs
                if not find_negatives(label_batch) and self.skip_batches:
                    continue
                pred_pose_batch = []
                pred_move_batch = []
                for pose_true, move_true in zip(pose_batch, move_batch):
                    pose_true, move_true = pose_true.to(device), move_true.to(device)
                    pose_pred, move_pred = self.model(pose_true, move_true)
                    pred_pose_batch.append(pose_pred)
                    pred_move_batch.append(move_pred)
                loss = self.loss_fn(pred_pose_batch, pred_move_batch, label_batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            test_losses = []
            self.model = self.model.eval()

            with torch.no_grad():
                for label_test_batch, pose_test_batch, move_test_batch in zip(iter(label_test_loader), iter(pose_test_loader), iter(move_test_loader)):
                    if not find_negatives(label_test_batch) and self.skip_batches:
                        continue
                    pred_pose_batch = []
                    pred_move_batch = []
                    for pose_true, move_true in zip(pose_test_batch, move_test_batch):
                        pose_true, move_true = pose_true.to(device), move_true.to(device)
                        pose_pred, move_pred = self.model(pose_true, move_true)
                        pred_pose_batch.append(pose_pred)
                        pred_move_batch.append(move_pred)
                    loss = self.loss_fn(pred_pose_batch, pred_move_batch, label_test_batch)
                    test_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            test_loss = np.mean(test_losses)
            print(f"Epoch {epoch}: train loss {train_loss}, test loss {test_loss}")
