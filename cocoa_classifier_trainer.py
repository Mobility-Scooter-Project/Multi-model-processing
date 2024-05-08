import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, RandomSampler
from cocoa_classifier import CocoaClassifier
from config import RANDOM_SEED, LEARNING_RATE
from utils import find_negatives, get_seq_label

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CocoaClassifierTrainer():
    def __init__(self, seq_len, batch_size, x_n_features, y_n_features, embedding_dim=16):
        self.model = CocoaClassifier(seq_len, x_n_features, y_n_features, embedding_dim)
        self.loss_fn = nn.BCELoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.batch_size = batch_size
    
    def load_encoder(self, PATH):
        self.model.load_encoder(PATH)

    def freeze_encoder(self, state):
        self.model.freeze_encoder(state)
    
    def get_model(self):
        return self.model

    def train(self, epochs, label_train_dataset, label_test_dataset, pose_train_dataset, pose_test_dataset, 
                move_train_dataset, move_test_dataset):
        G = torch.Generator()
        G.manual_seed(RANDOM_SEED)
        train_sampler = RandomSampler(data_source=label_train_dataset, generator=G)
        test_sampler = RandomSampler(data_source=label_test_dataset, generator=G)

        for epoch in range(0, epochs):
            model = self.model.train()
            train_losses = []
            train_sampler_save = list(train_sampler)
            test_sampler_save = list(test_sampler)

            label_train_loader = DataLoader(label_train_dataset, batch_size=self.batch_size, sampler=train_sampler_save)
            pose_train_loader = DataLoader(pose_train_dataset, batch_size=self.batch_size, sampler=train_sampler_save)
            move_train_loader = DataLoader(move_train_dataset, batch_size=self.batch_size, sampler=train_sampler_save)
            label_test_loader = DataLoader(label_test_dataset, batch_size=self.batch_size, sampler=test_sampler_save)
            pose_test_loader = DataLoader(pose_test_dataset, batch_size=self.batch_size, sampler=test_sampler_save)
            move_test_loader = DataLoader(move_test_dataset, batch_size=self.batch_size, sampler=test_sampler_save)

            for label_batch, pose_batch, move_batch in zip(iter(label_train_loader), iter(pose_train_loader), iter(move_train_loader)):
                # Skip batches without negative pairs
                if not find_negatives(label_batch):
                    continue
                for pose_true, move_true, label_true in zip(pose_batch, move_batch, label_batch):
                    pose_true, move_true, label_true = pose_true.to(device), move_true.to(device), label_true.to(device)
                    pred = model(pose_true, move_true)
                    true = torch.tensor(get_seq_label(label_true), dtype=torch.float32)
                    loss = self.loss_fn(pred, true)
                    loss.backward()
                    # Parameters are updated on every sample
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_losses.append(loss.item())

            test_losses = []
            model = self.model.eval()
            with torch.no_grad():
                for label_test_batch, pose_test_batch, move_test_batch in zip(iter(label_test_loader), iter(pose_test_loader), iter(move_test_loader)):
                    if not find_negatives(label_test_batch):
                        continue
                    for pose_true, move_true, label_true in zip(pose_test_batch, move_test_batch, label_test_batch):
                        pose_true, move_true = pose_true.to(device), move_true.to(device)
                        pred = model(pose_true, move_true)
                        true = torch.tensor(get_seq_label(label_true), dtype=torch.float32)
                        loss = self.loss_fn(pred, true)
                        test_losses.append(loss.item())
            train_loss = np.mean(train_losses)
            test_loss = np.mean(test_losses)
            print(f"Epoch {epoch}: train loss {train_loss}, test loss {test_loss}")