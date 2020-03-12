"""
"""
import os
from matplotlib import pyplot as plt


def visualize_result(history, save_dir=None):
    """Visualize accuracy and loss."""
    data = history.history
    plt.subplot(121)
    plt.plot(data['acc'], label='accuracy')
    plt.plot(data['val_acc'], label='val accuracy')
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')

    plt.subplot(122)
    plt.plot(data['loss'], label='loss')
    plt.plot(data['val_loss'], label='val loss')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    if save_dir:
        filepath = os.path.join(save_dir, 'train.png')
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()


def visualize_roc(fpr, tpr, threshold, acc, save_dir=None):
    """
    """
    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr)
    if save_dir:
        filepath = os.path.join(save_dir, 'roc_auc.png')
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()
