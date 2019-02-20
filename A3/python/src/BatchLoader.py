import numpy as np


class BatchLoader():
    def __init__(self, indices, batch_size, data, labels, shuffle=False):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        self.labels = labels

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        idx = 0
        while idx+self.batch_size <= len(self.indices):
            batch_idx = self.indices[idx:idx+self.batch_size]
            yield self.data[batch_idx], self.labels[batch_idx]
            idx += self.batch_size
        if idx < len(self.indices):
            batch_idx = self.indices[idx:]
            yield self.data[batch_idx], self.labels[batch_idx]

    def __len__(self):
        return int((len(self.indices) + self.batch_size - 1)/self.batch_size)
