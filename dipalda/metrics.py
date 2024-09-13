import random


def recall(tp, fn):
    value = 0
    if tp + fn != 0:
        value = float(tp) / (tp + fn)
    return value


def precision(tp, fp):
    value = 0
    if tp + fp != 0:
        value = float(tp) / (tp + fp)
    return value


def accuracy(tp, tn, fp, fn):
    value = 0
    if tp + tn + fp + fn != 0:
        value = float(tp + tn) / (tp + tn + fp + fn)
    return value


def fscore(tp, fp, fn):
    value = 0
    if tp + fp + fn != 0:
        value = float(2 * tp) / (2 * tp + fp + fn)
    return value
