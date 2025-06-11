import numpy as np


class Dataset:
    def __init__(self,
                 feature: np.ndarray,
                 label: np.ndarray,
                 unique: bool = True,
                 *args,
                 **kwargs):
        """


        :param feature: np.ndarray, feature of labeled data. Normally the index of a vectorized tensor.
        :param label: np.ndarray, label of labeled data. Normally, the value of corresponded feature.
        :param unique: bool, True by default, whether force the feature to be unique. Recommended to use `unique=True`
        :param args: arguments
        :param kwargs: keyword arguments
        """

        self.feature = feature
        self.label = label
        self.unique = unique

        self.size_check()
        self.flush()

    def __len__(self):
        self.size_check()
        return len(self._feature)

    def __getitem__(self, idx):
        self.size_check()
        return self._feature[idx], self._label[idx]

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
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, feature):
        self._feature = feature

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def unique(self):
        return self._unique

    @unique.setter
    def unique(self, unique):
        self._unique = unique

    def size_check(self):
        # assert len(self.feature) == len(self.label), 'label and data should have the same shape'
        return True

    def flush(self):
        if self.unique:
            _, index_uni = np.unique(self.feature, return_index=True, axis=0)
            self.feature = self.feature[index_uni]
            self.label = self.label[index_uni]

