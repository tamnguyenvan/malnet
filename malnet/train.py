"""Train our network on Ember dataset. Before running this script, you need to
extract features first. See create_data.py for the details.

Usage: python train.py \
        --data_dir DATA_DIR \
        --lr LEARNING_RATE \
        --batch_size BATCH_SIZE \
        --epochs EPOCHS
"""
import os
import argparse
import numpy as np
import pickle as pkl
from sys import argv

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, Activation
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score

import ember
import utils


# Fix tensorflow bug on rtx card
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))


def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='data',
                        help='Directory that stores our dataset.')
    parser.add_argument('--save_dir', dest='save_dir', type=str, default='saved_models',
                        help='Directory to save model.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                        help='Mini-batch samples.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Number of epochs.')
    parser.add_argument('--split', dest='split', type=float, default=0.1,
                        help='Validation dataset ratio.')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--scale', dest='scale', type=float, default=1.,
                        help='Scale of training/test dataset.')
    return parser.parse_args(argv)


# Parse arguments
args = parse_arguments(argv[1:])

# Hyperparameters
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
weight_decay = 5e-4
save_dir = args.save_dir

# Params
num_classes = 2
split = args.split


# Generate dummy data
print('Loading data...')
data_dir = args.data_dir
X_train, y_train, X_test, y_test = \
        ember.read_vectorized_features(data_dir, scale=args.scale)
X_train = np.array(X_train)
X_test = np.array(X_test)


# Only keep supervised data, leave unsupervised data
# Note that unsupervised data has label -1
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
print('Train/Val/Test: {}/{}/{}'.format(X_train.shape[0],
                                        X_val.shape[0],
                                        X_test.shape[0]))


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
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
    pkl.dump(scaler, f)


# Model expects 3D input, so expands the last dim.
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# For convenient
dim = X_train.shape[1]
regularizer = regularizers.l2(weight_decay)

# Build the model
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
print('Batch size: {}'.format(batch_size))
print('Epochs: {}'.format(epochs))
print('Learning rate: {}'.format(learning_rate))
print('Weight decay: {}'.format(weight_decay))

# Define the optimizer and compile model
optimizer = Adam(learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Training
model_name = 'malnet_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint])

# Visualize the result
utils.visualize_result(history, save_dir)

# ROC curve
y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(np.argmax(y_test, axis=1), y_pred[:, 1], pos_label=1)
acc = np.mean(np.equal(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
utils.visualize_roc(fpr, tpr, thresholds, acc, save_dir)
