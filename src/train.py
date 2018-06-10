# train.py

"""
- Author: tamnv
- Description: This script contains the entire training pipeline
of MalNet: The processing was followed steps:
vectorized data -> build model -> training -> evaluate ->
save model -> summary.

Usage: python train.py [--model all|malnet|et|rt]
[--batch-size BATCH_SIZE] [--epochs EPOCHS]
"""

import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle as pkl
from sys import argv

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, Activation
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc


import ember
from util import TrainValTensorBoard


def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('-m', '--model', dest='model', type=str, default='malnet',
                        help='Model will be used. The following params are avaible: '
                        'malnet, et, rf, all.')
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str, default='data',
                        help='Directory that stores our dataset.')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint', type=str, default='checkpoint',
                        help='Directory to save checkpoint.')
    parser.add_argument('-g', '--graph', dest='graph', type=str, default='graph',
                        help='Directory to save model graph.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='Mini-batch samples.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Number of epochs.')
    parser.add_argument('-v', '--visual', dest='visual', type=str, default='visualization',
                        help='Directory to save visualization.')
    parser.add_argument('--scale', dest='scale', type=float, default=1.,
                        help='Scale of training/test dataset.')
    return parser.parse_args(argv)


# Parse arguments
args = parse_arguments(argv[1:])

# Hyperparameters
batch_size = args.batch_size
epochs = args.epochs
learning_rate = 1e-4
weight_decay = 5e-4

# Params
num_classes = 2
split = 0.1


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
result_dir = os.path.join(root_dir, 'result')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


# Generate dummy data
print('Loading data...')
data_dir = os.path.join(root_dir, args.data_dir)
X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir, scale=args.scale)
X_train = np.array(X_train)
X_test = np.array(X_test)


# Remain only supervised data, leave unsupervised data
# unsupervised data has label -1
X_train = X_train[y_train != -1]
y_train = y_train[y_train != -1]
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

idx = int((1. - split) * X_train.shape[0])
X_val, y_val = X_train[idx:], y_train[idx:]
X_train, y_train = X_train[:idx], y_train[:idx]


X_test = X_test[y_test != -1]
y_test = y_test[y_test != -1]


# Convert labels to one-hot
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)


# Preprocessing data before training
# Save the standard scaler for deploying phase.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
checkpoint_dir = os.path.join(result_dir, args.checkpoint)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
with open(os.path.join(checkpoint_dir, 'scaler.pkl'), 'wb') as f:
    pkl.dump(scaler, f)


# Model expects 3D input, so expands the last dim.
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# For convenient
dim = X_train.shape[1]
regularizer = regularizers.l2(weight_decay)


result_dict = dict()
if args.model == 'malnet' or args.model == 'all':
    model = Sequential()
    model.add(Conv1D(128, 64, strides=64, activation='relu',
        kernel_regularizer=regularizer, input_shape=(dim, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 3, strides=2, kernel_regularizer=regularizer,activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=regularizer, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, kernel_regularizer=regularizer, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, kernel_regularizer=regularizer, activation='softmax'))
    model.summary()

    # Print verbose information
    print('Model: {}'.format(args.model))
    print('Batch size: {}'.format(batch_size))
    print('Epochs: {}'.format(epochs))
    print('Learning rate: {}'.format(learning_rate))
    print('Weight decay: {}'.format(weight_decay))
    print('Training/Val/Test samples: %d/%d/%d' % (X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    # Define optimizer and compile model
    optimizer = Adam(learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Training
    # Visualize model on tensorboard
    history = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=[TrainValTensorBoard()])

    # Summarize history for accuracy
    visual_dir = os.path.join(result_dir, args.visual)
    if not os.path.exists(visual_dir):
        os.mkdir(visual_dir)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    lower, higher = min(acc + val_acc), max(acc + val_acc)

    plt.plot(np.arange(epochs), acc)
    plt.plot(np.arange(epochs), val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(0, epochs + 1)
    plt.ylim(max(lower - 0.1 * (higher - lower), 0.), min(higher + 0.1 * (higher - lower), 1.))
    plt.xticks(np.arange(0, epochs, 5))
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(visual_dir, 'acc.png'))
    plt.close()

    # summarize history for loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    lower, higher = min(loss + val_loss), max(loss + val_loss)
    plt.plot(np.arange(epochs), loss)
    plt.plot(np.arange(epochs), val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, epochs + 1)
    plt.ylim(max(lower - 0.1 * (higher - lower), 0.), min(higher + 0.2 * (higher - lower), 1.))
    plt.xticks(np.arange(0, epochs, 5))
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(visual_dir, 'loss.png'))
    plt.close()

    # Save model graph to JSON
    model_json = model.to_json()
    with open(os.path.join(checkpoint_dir, 'malnet.json'), "w") as json_file:
        json_file.write(model_json)

    # Save weights for further use
    model.save(os.path.join(checkpoint_dir, 'malnet.h5'))
    print("Saved model to disk.")

    # ROC curve
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_pred[:, 1], pos_label=1)
    acc = np.mean(np.equal(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    result_dict['malnet'] = [fpr, tpr, thresholds, acc]

if args.model == 'et' or args.model == 'all':
    print('Training Random Forest...')
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=123)
    X_train = np.squeeze(X_train, axis=-1)
    X_test = np.squeeze(X_test, axis=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_pred[:, 1], pos_label=1)
    acc = np.mean(np.equal(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    result_dict['et'] = [fpr, tpr, thresholds, acc]

if args.model == 'rf' or args.model == 'all':
    print('Training Extra Trees...')
    clf = ExtraTreesClassifier(n_estimators=100, max_depth=2, random_state=123)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_pred[:, 1], pos_label=1)
    acc = np.mean(np.equal(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    result_dict['rf'] = [fpr, tpr, thresholds, acc]


# Summary results
for m, metrics in result_dict.items():
    fpr, tpr, threshold, acc = metrics
    print('Accuracy on test of %s: %.4f' % (m, acc))
    print('AUC of %s: %.4f' % (m, auc(fpr, tpr)))
    print('')
    plt.plot(fpr, tpr)

plt.xlim([-0.1, 1.05])
plt.ylim([-0.1, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend([model_name.upper() for model_name in list(result_dict.keys())], loc='lower right')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.savefig(os.path.join(visual_dir, 'roc.png'))
plt.close()


# Zoom
for m, metrics in result_dict.items():
    fpr, tpr, thresholds, acc = metrics
    plt.plot(fpr, tpr)

plt.xlim([-0.1, 1.05])
plt.ylim([0.6, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend([model_name.upper() for model_name in list(result_dict.keys())], loc='lower right')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.savefig(os.path.join(visual_dir, 'roc_zoom.png'))
plt.close()


# Log scale of ROC curve
fpr, tpr, thresholds, _ = result_dict['malnet']
tpr = tpr[fpr > 5e-5]
thresholds = thresholds[fpr > 5e-5]
fpr = np.log10(fpr[fpr > 5e-5])
plt.plot(fpr, tpr, color='black')

idx = np.argmin(np.abs(fpr + 3.))

print('Threshold: %.4f FPR: %.4f Detection rate: %.4f' %
    (thresholds[idx] if thresholds[idx] <= 1.0 else 1.0, np.power(10, fpr[idx]), tpr[idx]))
plt.plot([np.min(fpr) - 5, fpr[idx]], [tpr[idx], tpr[idx]], color='red', linestyle='--')
plt.plot([fpr[idx], fpr[idx]], [tpr[idx], 0], color='red', linestyle='--')

idx = np.argmin(np.abs(fpr + 2.))
print('Threshold: %.4f FPR: %.4f Detection rate: %.4f' %
    (thresholds[idx] if thresholds[idx] <= 1.0 else 1.0, np.power(10, fpr[idx]), tpr[idx]))

idx = np.argmin(np.abs(fpr + 1.))
print('Threshold: %.4f FPR: %.4f Detection rate: %.4f' %
    (thresholds[idx] if thresholds[idx] <= 1.0 else 1.0, np.power(10, fpr[idx]), tpr[idx]))


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([np.min(fpr), 0.])
plt.ylim([0.65 if args.scale == 1. else 0., 1.01])
plt.xticks([-4, -3, -2, -1, 0], [r'$10^{{-4}}$', r'$10^{{-3}}$', r'$10^{{-2}}$', r'$10^{{-1}}$', r'$10^{{0}}$'])
plt.grid(True)
plt.title('MalNet ROC Curve')
plt.savefig(os.path.join(visual_dir, 'roc_log_scale.png'))
plt.close()
