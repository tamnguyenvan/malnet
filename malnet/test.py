"""Using pretrained model to predict some real examples to demonstrate it works.

Usage: python test.py --input_file PE_FILE --model_path MODEL_PATH
"""
import os
import glob
import json
import argparse
import pickle as pkl
from sys import argv

import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model

import ember

# Fix tensorflow bug on rtx card
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('--input_file', dest='input_file', type=str,
                        help='Path to PE file.')
    parser.add_argument('--model_path', dest='model_path', type=str,
                        help='Path to model directory.')
    parser.add_argument('--scaler_path', dest='scaler_path', type=str,
                        help='Path to the scaler object file.')
    parser.add_argument('--threshold', dest='threshold', type=float, default=0.273,
                        help='Threshold to distinguish benign and malicous.')
    return parser.parse_args(argv)

# Parse args
args = parse_arguments(argv[1:])
input_file = args.input_file
model_path = args.model_path

print('Example: %s' % input_file)
print('Model path: %s' % args.model_path)
print('Threshold: %f' % args.threshold)

# Extract features from PE file
extractor = ember.features.PEFeatureExtractor()
with open(input_file, 'rb') as f:
    raw_bytes = f.read()
feature = np.array(extractor.feature_vector(raw_bytes), dtype=np.float32)

# Load model and predict
print('Loading model from {}'.format(model_path))
scaler_path = args.scaler_path
with open(scaler_path, 'rb') as f:
    scaler = pkl.load(f)
model = load_model(model_path)

features = np.array([feature], dtype=np.float32)
features = scaler.transform(features)
features = np.expand_dims(features, axis=-1)

score = model.predict(features)[0]
if score[-1] < args.threshold:
    print('Score: %.5f -> Benign' % score[-1])
else:
    print('Score: %.5f -> Malicous' % score[-1])
