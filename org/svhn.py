import numpy as np
import scipy.io as sio
from random import random


class SVHN:
    def __init__(self, file_path, n_classes, gray=False):
        self.n_classes = n_classes

        # # Load Train & Validation Set
        train = sio.loadmat(file_path + "/train_32x32.mat")
        self.train_validation_labels = self.__one_hot_encode(train['y'])
        self.train_validation_examples = train['X'].shape[3]
        self.train_validation_data = self.__store_data(train['X'].astype("float32"), self.train_validation_examples,
                                                       gray)

        self.train_labels = self.train_validation_labels[0:int(0.7 * self.train_validation_examples)]
        self.train_examples = int(train['X'].shape[3] * 0.7)
        self.train_data = self.train_validation_data[0:int(0.7 * self.train_validation_examples)]

        self.validation_labels = self.train_validation_labels[int(0.7 * self.train_validation_examples) + 1:]
        self.validation_examples = int(train['X'].shape[3] * 0.3)
        self.validation_data = self.train_validation_data[int(0.7 * self.train_validation_examples) + 1:]

        # Load Test Set
        test = sio.loadmat("../res/test_32x32.mat")
        self.test_labels = self.__one_hot_encode(test['y'])
        self.test_examples = test['X'].shape[3]
        self.test_data = self.__store_data(test['X'].astype("float32"), self.test_examples, gray)

    def __one_hot_encode(self, data):
        n = data.shape[0]
        one_hot = np.zeros(shape=(data.shape[0], self.n_classes))
        for s in range(n):
            temp = np.zeros(self.n_classes)

            num = data[s][0]
            if num == 10:
                temp[0] = 1
            else:
                temp[num] = 1

            one_hot[s] = temp

        return one_hot

    def __store_data(self, data, num_of_examples, gray):
        d = []

        for i in range(num_of_examples):
            if gray:
                d.append(self.__rgb2gray(data[:, :, :, i]))
            else:
                d.append(data[:, :, :, i])

        return np.asarray(d)

    def __rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
