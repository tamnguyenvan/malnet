"""
- Author: tamnv
- Description: This script will extract raw data from EMBER
json files, then write into 4 files: X_train.dat, X_test.dat,
y_train.dat and y_test.dat
"""

import argparse
from sys import argv

import ember


def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data',
                        help='Path to data directory.')
    parser.add_argument('--scale', dest='scale', type=float, default=1.,
                        help='Scale of training/test dataset.')
    return parser.parse_args(argv)


# Parse arguments
args = parse_arguments(argv[1:])
data_dir = args.data_dir

ember.create_vectorized_features(data_dir, scale=args.scale)
