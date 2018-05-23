# train.py

"""
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

import ember
from util import TrainValTensorBoard


def parse_arguments(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='MalNet')
    parser.add_argument('-m', '--model', dest='model', type=str, default='malnet',
                        help='Model will be used. The following params are avaible: '
                        'malnet, ert, rf.')
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str, default='data',
                        help='Directory that stores our dataset.')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint', type=str, default='checkpoint',
                        help='Directory to save checkpoint.')
    parser.add_argument('-g', '--graph', dest='graph', type=str, default='graph',
                        help='Directory to save model graph.')
    parser.add_argument('-v', '--visual', dest='visual', type=str, default='visualization',
                        help='Directory to save visualization.')
    return parser.parse_args(argv)


# Parse arguments
args = parse_arguments(argv[1:])

# Hyperparameters
batch_size = 512
epochs = 20
learning_rate = 1e-3
weight_decay = 5e-4

# Params
num_classes = 2


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
result_dir = os.path.join(root_dir, 'result')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


# Generate dummy data
print('Loading data...')
data_dir = os.path.join(root_dir, args.data_dir)
X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)
X_train = np.array(X_train)
X_test = np.array(X_test)


# Remain only supervised data, leave unsupervised data
# unsupervised data has label -1
X_train = X_train[y_train != -1]
y_train = y_train[y_train != -1]
X_test = X_test[y_test != -1]
y_test = y_test[y_test != -1]


# Model expects 3D input, so expands the last dim.
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# Convert labels to one-hot
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)


# Preprocessing data before training
# Save the standard scaler for deploying phase.
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
checkpoint_dir = os.path.join(result_dir, args.checkpoint)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
with open(os.path.join(checkpoint_dir, 'scaler.pkl'), 'wb') as f:
    pkl.dump(scaler, f)


# For convenient
dim = X_train.shape[1]
regularizer = regularizers.l2(weight_decay)


# Build model graph
if args.model == 'malnet':
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
elif args.model == 'ert':
    pass
elif args.model == 'rf':
    pass


# Print verbose information
print('Model: {}'.format(args.model))
print('Training: {} {}'.format(X_train.shape, y_train.shape))
print('Testing: {} {}'.format(X_test.shape, y_test.shape))
print('Batch size: {}'.format(batch_size))
print('Epochs: {}'.format(epochs))
print('Learning rate: {}'.format(learning_rate))
print('Weight decay: {}'.format(weight_decay))


# Define optimizer and compile model
optimizer = Adam(learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Training
# Visualize model on tensorboard
history = model.fit(X_train, y_train,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[TrainValTensorBoard()])

# Summarize history for accuracy
visual_dir = os.path.join(result_dir, args.visual)
if not os.path.exists(visual_dir):
    os.mkdir(visual_dir)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(visual_dir, 'acc.png'))
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(visual_dir, 'loss.png'))
plt.close()

# Save model graph to JSON
model_json = model.to_json()
with open(os.path.join(checkpoint_dir, args.model + '.json'), "w") as json_file:
    json_file.write(model_json)

# Save weights for further use
model.save(os.path.join(checkpoint_dir, args.model + '.h5'))
print("Saved model to disk.")


