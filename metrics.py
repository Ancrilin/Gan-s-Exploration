# coding: utf-8
# @author: Ross
# @file: metrix.py 
# @time: 2020/01/15
# @contact: devross@gmail.com
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def binary_accuracy(predicts, labels) -> float:
    """
    compute binary accuracy
    :param predicts: [num, 1]
    :param labels: [num, 1]
    :param threshold: float
    :return:
    """
    return accuracy_score(labels, predicts)


def binary_recall_fscore(predicts, labels):
    """
    compute binary accuracy
    :param predicts: [num, 1]
    :param labels: [num, 1]
    :param threshold: (recall, fscore)
    :return:
    """

    return precision_recall_fscore_support(labels, predicts)


def ind_class_accuracy(predicts: torch.Tensor, labels: torch.Tensor, oos_index=0):
    """忽略oos的正确率"""
    _labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    _predicts = predicts.numpy() if isinstance(predicts, torch.Tensor) else predicts

    sample_weight = (_labels != oos_index).astype(int)
    return accuracy_score(_labels, _predicts, sample_weight=sample_weight)


def accuracy(predicts: torch.Tensor, labels: torch.Tensor):
    """正确率"""
    _labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    _predicts = predicts.numpy() if isinstance(predicts, torch.Tensor) else predicts

    return accuracy_score(_labels, _predicts)


def binary_classification_report(predicts, labels):
    '''
    输出precision, accuracy, f1-score, support
    :param predicts: 预测标签
    :param labels: 真实标签
    :return: report
    '''
    predicts = predicts.numpy() if isinstance(predicts, torch.Tensor) else predicts
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    return classification_report(labels, predicts, labels=[0, 1], target_names=['ood', 'ind'])


def plot_confusion_matrix(y_true, y_pred, save_path, classes=['ood', 'ind'], cmap=plt.cm.Blues):
    """Plot a confusion matrix using ground truth and predictions."""
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #  Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Axis
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Values
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Display
    plt.savefig(save_path + '/confusion_matrix.png')
    plt.show()


def cal_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer
