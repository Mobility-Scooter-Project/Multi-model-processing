import torch
from cocoa_classifier import CocoaClassifier
from config import POSE_N_FEATURES, MOVE_N_FEATURES
import unittest

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SEQUENCE_LENGTH = 6

class TestCocoaClassifierMethods(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model = CocoaClassifier(SEQUENCE_LENGTH, POSE_N_FEATURES, MOVE_N_FEATURES, embedding_dim=16)
        self.model.to(device)

    def test_encoder_freeze(self):
        for name, param in self.model.named_parameters():
            self.assertTrue(param.requires_grad)
        self.model.freeze_encoder(True)
        for name, param in self.model.named_parameters():
            self.assertFalse(param.requires_grad)
        
if __name__ == '__main__':
    unittest.main()