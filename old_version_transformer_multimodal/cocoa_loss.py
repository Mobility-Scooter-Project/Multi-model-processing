import torch
import torch.nn as nn
from config import THRESHOLD

class CocoaLoss(nn.Module):
    def __init__(self, tau, lam):
        super(CocoaLoss, self).__init__()
        self.tau = tau
        self.lam = lam

    def forward(self, x_pred_batch, y_pred_batch, label_batch):
        pos_error = self.__calc_pos_error(x_pred_batch, y_pred_batch)
        neg_idxs = self.__find_negatives(label_batch)
        x_neg_error = self.__calc_neg_error(x_pred_batch, neg_idxs)
        y_neg_error = self.__calc_neg_error(y_pred_batch, neg_idxs)
        return  pos_error+self.lam*(x_neg_error + y_neg_error)

    def __calc_pos_error(self, x_pred_batch, y_pred_batch):
        pos_error = 0
        for pose, move in zip(x_pred_batch, y_pred_batch):
            """
            print("Pose shape:", pose.shape)
            print("Move shape:", move.shape)
            """
            corr = torch.linalg.matmul(pose, move)
            corr = torch.div(corr, torch.mul(torch.linalg.vector_norm(pose), torch.linalg.vector_norm(move)))
            corr = torch.sub(1, corr)
            corr = torch.exp(corr / self.tau)
            pos_error = torch.add(pos_error, corr)
        return pos_error / len(x_pred_batch)

    def __find_negatives(self, label_batch):
        negatives = []
        for idx, label_seq in enumerate(label_batch):
            i = 0
            for label in label_seq:
                if not label:
                    i += 1
            if i > THRESHOLD:
                negatives.append(idx)
        return negatives

    def __calc_neg_error(self, pred_batch, neg_indexs):
        neg_error = 0
        if not neg_indexs:
            return neg_error
        i = 0
        for idx in neg_indexs:
            non_neg_sequences = [seq for i, seq in enumerate(pred_batch) if i not in neg_indexs]
            neg_l2norm = torch.linalg.vector_norm(pred_batch[idx])
            for seq in non_neg_sequences:
                discrim = torch.linalg.matmul(pred_batch[idx], seq)
                discrim = torch.div(discrim, torch.mul(neg_l2norm, torch.linalg.vector_norm(seq)))
                discrim = torch.exp(discrim / self.tau)
                neg_error = torch.add(neg_error, discrim)
                i += 1
        return neg_error / i
