import numpy as np
from .dataset import Dataset


class Dataloader:
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 10,
                 shuffle: bool = True,
                 sort: bool = True):
        """

        :param dataset: Dataset instance, predefined dataset of optim
        :param batch_size: int, the batch size to be used in the training
        :param shuffle: bool, whether to shuffle the dataset.
        :param sort: bool, whether to sort the features (index of tensor) in dataset. Commanded to be True since the
        returned value of gradient of single entry functions in the nuwa.util must be sorted.
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sort = sort

        self.sets = []
        self.current = 0
        self.init()

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        index = self.sets[idx]
        if self.sort:
            index_sort = np.sort(index, axis=0).reshape(-1, )
            return self.dataset[index_sort]
        else:
            return self.dataset[index]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        else:
            self.current += 1
            return self[self.current - 1]

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool):
        self._shuffle = shuffle

    def init(self):
        n = len(self.dataset)

        index = np.linspace(0, n - 1, n, dtype=int)

        if self.shuffle:
            index = np.random.choice(range(len(self.dataset)), len(self.dataset), replace=False)

        if self.batch_size >= n:
            self.sets = [index]
        elif self.batch_size < n:
            self.sets = np.array_split(index, n // self.batch_size)
        else:
            raise "Batch size should be integer larger than 0"

