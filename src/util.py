# util.py

"""
- Author: tamnv
- Description: This module just contains utility functions
for the training script.
Metrics class will be used to add validation measurements
during training.
TrainValTensorBoard class for summary during training.
"""

import os
import glob
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from keras.callbacks import Callback, TensorBoard


class Metrics(Callback):
    """
    """
    def on_train_begin(self, logs={}):
        """
        """
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        """
        """
        val_predict = (np.asarray(self.model.predict(
            self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('val_precision: %f — val_recall %f' %
              (_val_precision, _val_recall))


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def get_paths(model_dir):
    """Helper function to get all model paths."""
    path_dict = dict()
    for root, _, filenames in os.walk(model_dir):
        for filename in filenames:
            path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.json':
                path_dict['graph'] = path
            elif ext == '.h5':
                path_dict['model'] = path
            elif ext == '.pkl':
                path_dict['scaler'] = path
    return path_dict