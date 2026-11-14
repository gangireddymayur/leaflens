"""Classification metrics in plain NumPy."""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def per_class_f1(cm):
    """F1 for each class from a confusion matrix (rows = true, columns = predicted)."""
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(cm):
    """Mean F1 over classes that actually occur in the true labels."""
    present = cm.sum(axis=1) > 0
    return float(per_class_f1(cm)[present].mean()) if present.any() else 0.0
