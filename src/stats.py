# stats.py

"""
- Author: tamnv
- Description: This script process some statistics on input
data for further understanding about the EMBER dataset.

Usage: python stats.py --data-dir [path/to/dataset/dir]
"""

import os
import random
import numpy as np
import argparse
import ember
from sklearn.preprocessing import StandardScaler

from sys import argv


def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str, default='data',
                        help='Directory that stores our dataset.')
    return parser.parse_args(argv)


args = parse_arguments(argv[1:])
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, args.data_dir)

print('Loading data...')
X_train, y_train, _, _ = ember.read_vectorized_features(data_dir)
X_train = np.array(X_train)


# Remain only supervised data, leave unsupervised data
# unsupervised data has label -1
X_train = X_train[y_train != -1]

k = 5
indices = random.sample(list(range(X_train.shape[1])), k)
select_train = X_train[:, indices]

# Statistics
max_train = np.max(select_train, axis=0)
min_train = np.min(select_train, axis=0)
mean_train = np.mean(select_train, axis=0)
std_train = np.std(select_train, axis=0)

print('Max: {}'.format(max_train))
print('Min: {}'.format(min_train))
print('Mean: {}'.format(mean_train))
print('Std: {}'.format(std_train))


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
normed_select_train = X_train[:, indices]
max_train = np.max(normed_select_train, axis=0)
min_train = np.min(normed_select_train, axis=0)
mean_train = np.mean(normed_select_train, axis=0)
std_train = np.std(normed_select_train, axis=0)

print('')
print('*' * 20)
print('')

print('Max: {}'.format(max_train))
print('Min: {}'.format(min_train))
print('Mean: {}'.format(mean_train))
print('Std: {}'.format(std_train))
