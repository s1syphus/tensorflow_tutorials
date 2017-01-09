"""
CIFAR10 dataset
"""

from dataset import Dataset


class CIFAR10Data(Dataset):
    def __init__(self, subset, data_dir):
        super(CIFAR10Data, self).__init__('CIFAR10', subset, data_dir)

    def num_classes(self):
        return 10

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return 50000
        if self.subset == 'validation':
            return 300000

