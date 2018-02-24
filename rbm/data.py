import numpy as np

class DataLoaderIter():
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.size = len(data)
        self.start = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.start >= self.size:
            raise StopIteration
        start = self.start
        end = min(self.start + self.batch_size, self.size)
        self.start = end
        return self.data[start : end]

class DataLoader(object):
    def __init__(self, data, batch_size, shuffle = False):
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.size = len(data)
        self.start = 0

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data)
        return DataLoaderIter(self.data, self.batch_size)