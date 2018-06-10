# eval.py

"""
- Author: tamnv
- Description: Evaluate pretrained model independently. Note:
we must have pretrained model before.

Usage: python eval.py --model [path/to/model/directory]
--data-dir [path/to/data/directory]
"""

import argparse
from sys import argv
import pickle as pkl

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import keras
from keras.models import model_from_json
import ember
from util import get_paths
from sklearn.metrics import roc_curve, auc


def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data',
                        help='Path to data directory contains test dataset.')
    parser.add_argument('--model', dest='model_dir', type=str,
                        help='Path to model directory.')
    parser.add_argument('--scale', dest='scale', type=float, default=1.,
                        help='Scale of training/test dataset.')
    return parser.parse_args(argv)


# Parse arguments
args = parse_arguments(argv[1:])

# Generate dummy data
print('Loading data...')
data_dir = args.data_dir
_, _, X_test, y_test = ember.read_vectorized_features(data_dir, scale=args.scale)
X_test = np.array(X_test)

X_test = X_test[y_test != -1]
y_test = y_test[y_test != -1]

model_dir = args.model_dir
path_dict = get_paths(model_dir)
json_file = open(path_dict['graph'], 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(path_dict['model'])
with open(path_dict['scaler'], 'rb') as f:
    scaler = pkl.load(f)

X_test = scaler.transform(X_test)
X_test = np.expand_dims(X_test, axis=-1)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

# ROC curve
y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_pred[:, 1], pos_label=1)
acc = np.mean(np.equal(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

# Summary result
print('Model directory: %s' % args.model_dir)
print('Accuracy: %.4f' % (acc))
print('AUC: %.4f' % (auc(fpr, tpr)))
print('')
plt.plot(fpr, tpr)

plt.xlim([-0.1, 1.05])
plt.ylim([-0.1, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend('MalNet', loc='lower right')
plt.title('ROC curve of MalNet')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()
plt.close()
