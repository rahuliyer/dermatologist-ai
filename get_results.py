import itertools
import os
import sys

import numpy as np
import pandas as pd

if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


def _finish_plot(save_path):
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    backend = (plt.get_backend() or "").lower()
    if "agg" in backend:
        plt.close()
    else:
        plt.show()


def _column_values(frame, columns):
    subset = frame[list(columns)]
    if hasattr(subset, "to_numpy"):
        return subset.to_numpy()
    return subset.values


def plot_roc_auc(y_true, y_pred, save_path=None):
    """
    This function plots the ROC curves and provides the scores.
    """

    # initialize dictionaries and array
    fpr = dict()
    tpr = dict()
    roc_auc = np.zeros(3)

    # prepare for figure
    plt.figure()
    colors = ['aqua', 'cornflowerblue']

    # for both classification tasks (categories 1 and 2)
    for i in range(2):
        # obtain ROC curve
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        # obtain ROC AUC
        roc_auc[i] = auc(fpr[i], tpr[i])
        # plot ROC curve
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label='ROC curve for task {d} (area = {f:.2f})'.format(d=i + 1, f=roc_auc[i]))
    # get score for category 3
    roc_auc[2] = np.average(roc_auc[:2])

    # format figure
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    _finish_plot(save_path)

    # print scores
    for i in range(3):
        print('Category {d} Score: {f:.3f}'.format(d=i + 1, f=roc_auc[i]))
    return roc_auc


def plot_confusion_matrix(y_true, y_pred, thresh, classes, save_path=None):
    """
    This function plots the (normalized) confusion matrix.
    """

    # obtain class predictions from probabilities
    y_pred = (y_pred >= thresh) * 1
    # obtain (unnormalized) confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    _finish_plot(save_path)


if __name__ == "__main__":

    preds_path = sys.argv[1]
    if len(sys.argv) == 3:
        thresh = float(sys.argv[2])
    else:
        thresh = 0.5

    # get ground truth labels for test dataset
    truth = pd.read_csv('ground_truth.csv')
    y_true = _column_values(truth, ["task_1", "task_2"])

    # get model predictions for test dataset
    y_pred = pd.read_csv(preds_path)
    y_pred = _column_values(y_pred, ["task_1", "task_2"])

    # plot ROC curves and print scores
    plot_roc_auc(y_true, y_pred, save_path="roc_curves.png")
    # plot confusion matrix
    classes = ['benign', 'malignant']
    plot_confusion_matrix(y_true[:, 0], y_pred[:, 0], thresh, classes, save_path="confusion_matrix.png")
