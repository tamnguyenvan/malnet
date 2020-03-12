"""Evaluate pretrained model independently. Note that we must have pretrained
model before.

Usage: python eval.py \
        --model_path MODEL_PATH \
        --scaler_path SCALER_PATH \
        --data_dir DATA_DIR \
"""
import argparse
from sys import argv
import pickle as pkl

import keras
import tensorflow as tf
import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_curve, auc

import utils
import ember

# Fix tensorflow bug on rtx card
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data',
                        help='Path to data directory contains test dataset.')
    parser.add_argument('--model_path', dest='model_path', type=str,
                        help='Path to model directory.')
    parser.add_argument('--scaler_path', dest='scaler_path', type=str,
                        help='Path to the scaler object file.')
    parser.add_argument('--scale', dest='scale', type=float, default=1.,
                        help='Scale of training/test dataset.')
    return parser.parse_args(argv)


# Parse arguments
args = parse_arguments(argv[1:])
model_path = args.model_path
scaler_path = args.scaler_path
num_classes = 2

print('Loading data...')
data_dir = args.data_dir
_, _, X_test, y_test = ember.read_vectorized_features(data_dir, scale=args.scale)
X_test = np.array(X_test)

# Only keep supervised data
# Note that unsupervised data has label -1
X_test = X_test[y_test != -1]
y_test = y_test[y_test != -1]

print('Loading model from {}'.format(model_path))
scaler_path = args.scaler_path
with open(scaler_path, 'rb') as f:
    scaler = pkl.load(f)
model_path = args.model_path
model = load_model(model_path)
model.summary()

X_test = scaler.transform(X_test)
X_test = np.expand_dims(X_test, axis=-1)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

# ROC curve
y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_pred[:, 1], pos_label=1)
acc = np.mean(np.equal(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
utils.visualize_roc(fpr, tpr, thresholds, acc)
