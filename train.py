import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Argument to train KNN')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--d-mode', type=str, default='Euclide', choices=['Euclide', 'Manhattan'], help='distance used in the algorithm')
args = parser.parse_args()

assert args.seed > 0, "The random seed must be a positive integer."

K_list = [3, 5, 7]
TESTING_RATIO = 0.2  # 20% of training data for testing


# read the data
data = pd.read_csv("data/breast-cancer-wisconsin.data").values
data = np.delete(data, 0, axis=1)   # remove the index in first column
# TODO[]: find the way to fill  the missing value
data = np.delete(data, 5, axis=1)   # remove the 5th column, which has incomplete value (in some line, it is filled with '?')
np.random.seed(args.seed)
np.random.shuffle(data)

data_test = data[np.arange(int(TESTING_RATIO*data.shape[0]))]
data_train = data[np.arange(int(TESTING_RATIO*data.shape[0]), data.shape[0])]
num_features = data_train.shape[1]-1
print('Training size: ', data_train.shape[0])
print('Testing size: ', data_test.shape[0])


def distance(x, y, mode='Euclide'):
    """
    Calculate the distance between two points.
    :param x: (np.array)
    :param y: (np.array)
    :param mode: (str)
    :return:
    """
    assert mode in ['Euclide','Manhattan'], 'choose a right distance formula'
    if mode == 'Euclide':
        return np.linalg.norm(x-y)
    elif mode == 'Manhattan':
        return np.sum(np.absolute(x-y))


def train():
    """
    Train KNN algorithm.

    :return:
    """
    accuracy_dict = {}
    for k in K_list:
        confusion_mat = np.zeros((2, 2))  # only two class: "2" and "4"
        for i in range(data_test.shape[0]):
            list_class_dis = []
            for j in range(data_train.shape[0]):
                d = distance(data_test[i][range(num_features)], data_train[j][range(num_features)], mode=args.d_mode)
                list_class_dis.append([data_train[j][-1], d])
            arr_class_dis = np.asarray(list_class_dis)
            predicted_class = np.bincount(arr_class_dis[:, 0][range(k)].astype(int)).argmax()
            # fill the confusion matrix
            if data_test[i][-1] == 2:
                if predicted_class == data_test[i][-1]:
                    confusion_mat[0, 0] += 1
                else:
                    confusion_mat[0, 1] += 1
            else:
                if predicted_class == data_test[i][-1]:
                    confusion_mat[1, 1] += 1
                else:
                    confusion_mat[1, 0] += 1
        # accuracy
        print(confusion_mat)
        acc = (confusion_mat[0, 0]+confusion_mat[1, 1])/np.sum(confusion_mat)
        accuracy_dict.update({k: acc})
    print("Dictionary of accuracy: ", accuracy_dict)
    print("Distance used: ", args.d_mode)


if __name__ == "__main__":
    train()

