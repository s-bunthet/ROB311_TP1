import numpy as np
import pandas as pd
K = 3
TESTING_SIZE = 0.2  # 20% of training data for validation
# accuracy bound
best_accuracy = 0
# read the data
data = pd.read_csv("data/breast-cancer-wisconsin.data", delimiter=',').values
data = np.delete(data, 0, axis=1)   # remove the index in first column
print(data[0][3])
np.random.shuffle(data)

data_test = data[np.arange(int(TESTING_SIZE*data.shape[0]))]
data_train = data[np.arange(int(TESTING_SIZE*data.shape[0]), data.shape[0])]


# def distance(x, y, mode='euclide'):
#     """
#
#     :param x:
#     :param y:
#     :param mode:
#     :return:
#     """
#
#     return np.lin
#
#
# def train():
#     """
#
#     :return:
#     """


# data_train =



# calculate confusion matrix
# calculate accuracy
# plot confusion matrix
# split the data (for training and testing)
# train