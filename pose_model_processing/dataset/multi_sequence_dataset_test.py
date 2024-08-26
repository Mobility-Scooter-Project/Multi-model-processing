import torch
import numpy as np
from multi_sequence_dataset import MultiSequenceDataset
import unittest

class TestMultiSequenceDataset(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.mockArr1 = np.array([[1,2,3],[4,5,6], [7,8,9], [10,11,12]])
        self.mockArr2 = np.array([[13,14,15], [16,17,18], [19,20,21], [22,23,24]])

    def test_dataset(self):
        SEQUENCE_LENGTH = 3
        expected_out = [[[1,2,3],[4,5,6], [7,8,9]],[[4,5,6], [7,8,9], [10,11,12]], [[13,14,15], 
                        [16,17,18], [19,20,21]], [[16,17,18], [19,20,21], [22,23,24]]]
        data_set = MultiSequenceDataset(SEQUENCE_LENGTH)
        data_set.add(self.mockArr1)
        for i in range(data_set.size):
            for j in range(len(data_set[i])):
                self.assertListEqual(list(data_set[i][j]), expected_out[i][j])
        data_set.add(self.mockArr2)
        for i in range(data_set.size):
            for j in range(len(data_set[i])):
                self.assertListEqual(list(data_set[i][j]), expected_out[i][j])
        
if __name__ == '__main__':
    unittest.main()