import numpy as np
import pandas as pd
import time
import argparse
from progress.bar import Bar

parser = argparse.ArgumentParser(description='Argument to train KNN')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--d-mode', type=str, default='Euclide', choices=['Euclide', 'Manhattan'], help='distance used in the algorithm')
args   = parser.parse_args()

assert args.seed >= 0, "The random seed must be a positive integer."

K_list        = [3, 5, 7, 9, 11, 13, 15]
TESTING_RATIO = 0.2  # 20% of training data for testing


## read the data
#data = pd.read_csv("data/breast-cancer-wisconsin.data").values
#data = np.delete(data, 0, axis=1)   # remove the index in first column
#data = np.delete(data, 5, axis=1)   # remove the 5th column, which has incomplete value (in some line, it is filled with '?')

### DATA IMPORTING (INTO AN ARRAY NAMED DATA)

#open the file
f = open("./data/breast-cancer-wisconsin.data", "r")

#read the data
raw_data      = f.read()
replaced_data = raw_data.replace('?', '-1')
lines         = replaced_data.split('\n')
num_lines     = len(lines)-1 #the last line is an empty string
num_col       = 11

#creation of an array containing the data
data = np.zeros((num_lines, num_col), dtype=int)

for k in range(num_lines):
    line_array = np.array(lines[k].split(',')).astype(int)
    data[k]    = line_array

# replacement of the invalid data
data_valid     = np.heaviside(data + 1, 0)
data_averages  = np.average(data, axis=0, weights=data_valid)
for k in range(num_col):
    data[:, k] = np.where(data[:, k]==-1, data_averages[k], data[:, k])

data = np.delete(data, 0, axis=1)   # remove the index in first column

np.random.seed(args.seed)
np.random.shuffle(data)

data_test    = data[np.arange(int(TESTING_RATIO*data.shape[0]))]
data_train   = data[np.arange(int(TESTING_RATIO*data.shape[0]), data.shape[0])]
num_features = data_train.shape[1]-1
print('Training size: ', data_train.shape[0])
print('Testing size: ',  data_test.shape[0])


def distance(x, y, mode='Euclide'):
    """
    Calculate the distance between two points.
    :param x: (np.array)
    :param y: (np.array)
    :param mode: (str)
    :return:
    """
    assert mode in ['Euclide','Manhattan'], 'choose a right distance formula'
    if mode   == 'Euclide':
        return np.linalg.norm(x-y)
    elif mode == 'Manhattan':
        return np.sum(np.absolute(x-y))


def train():
    """
    Train KNN algorithm.

    :return:
    """

    start_training_time = time.time()
    accuracy_dict       = {}

    # confusion matrix
    confusion_mat = np.zeros((len(K_list), 2, 2))  # only two class: "2" and "4"
    bar           = Bar('Training', max=data_test.shape[0])
    for i in range(data_test.shape[0]):
        list_class_distance = []
        for j in range(data_train.shape[0]):
            d = distance(data_test[i][range(num_features)], data_train[j][range(num_features)], mode=args.d_mode)
            list_class_distance.append((data_train[j][-1], d))
        arr_class_distance = np.array(list_class_distance, dtype=[('class', int), ('distance', np.float64)])
        arr_class_distance = np.sort(arr_class_distance, order='distance')
        for l in range(len(K_list)):
            predicted_class = np.bincount(arr_class_distance['class'][range(K_list[l])].astype(int)).argmax()
            # fill the confusion matrix
            if data_test[i][-1] == 2:
                if predicted_class == data_test[i][-1]:
                    confusion_mat[l, 0, 0] += 1
                else:
                    confusion_mat[l, 0, 1] += 1
            else:
                if predicted_class == data_test[i][-1]:
                    confusion_mat[l, 1, 1] += 1
                else:
                    confusion_mat[l, 1, 0] += 1
        bar.next()
    bar.finish()
    # accuracy
    acc = (confusion_mat[:, 0, 0]+confusion_mat[:, 1, 1])/np.sum(confusion_mat, axis=(1, 2))
    for l in range(len(K_list)):
        accuracy_dict.update({K_list[l]: '{0:.4f}'.format(acc[l])})

    print('Training time: {0:.4f} seconds'.format(time.time()-start_training_time))
    print("Dictionary of accuracy: ", accuracy_dict)
    print("Distance used: ", args.d_mode)


if __name__ == "__main__":
    train()

