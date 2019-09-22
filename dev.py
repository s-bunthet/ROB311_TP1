import numpy as np
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser(description='Argument to train KNN')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--d-mode', type=str, default='Euclide', choices=['Euclide', 'Manhattan'], help='distance used in the algorithm')
args = parser.parse_args()

assert args.seed >= 0, "The random seed must be a positive integer."

K_list = [3, 5, 7, 9, 11, 13, 15]
TESTING_RATIO = 0.2  # 20% of training data for testing


### Read the data ###

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

