# test.py

"""
- Author: tamnv
- Description: Use pretrained model to predict some examples
in real to demonstrate its performance.
"""

import os
import glob
import json
import argparse
import pickle as pkl
from sys import argv

import numpy as np
from keras.models import model_from_json

import ember
from util import get_paths


def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('-i', '--input', dest='input', type=str,
                        help='Path to PE file.')
    parser.add_argument('--model', dest='model', type=str,
                        help='Path to model directory.')
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.466,
                        help='Threshold to distinguish benign and malicous.')
    return parser.parse_args(argv)

# Parse args
args = parse_arguments(argv[1:])
input_file = args.input
print('Example: %s' % input_file)
print('Model dir: %s' % args.model)
print('Threshold: %f' % args.threshold)

# Extract features from PE file
extractor = ember.features.PEFeatureExtractor()
with open(input_file, 'rb') as f:
    raw_bytes = f.read()
feature = np.array(extractor.feature_vector(raw_bytes), dtype=np.float32)

# Load model and predict
print('Loading model...')
model_dir = args.model
path_dict = get_paths(model_dir)
json_file = open(path_dict['graph'], 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(path_dict['model'])
with open(path_dict['scaler'], 'rb') as f:
    scaler = pkl.load(f)

features = np.array([feature], dtype=np.float32)
features = scaler.transform(features)
features = np.expand_dims(features, axis=-1)

score = model.predict(features)[0]
if score[-1] < args.threshold:
    print('Score: %.5f -> Benign' % score[-1])
else:
    print('Score: %.5f -> Malicous' % score[-1])
