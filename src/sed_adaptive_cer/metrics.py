"""Pixel counts for semantic IoU and aggregation across images."""

import numpy as np


def confusion_matrix(target, predicted, num_classes):
    """Aggregate valid pixels in a [ground truth, prediction] matrix."""
    valid = target != 255
    encoded = num_classes * target[valid].astype(np.int64) + predicted[valid]
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def pixel_counts(confusion):
    return np.stack((confusion.sum(axis=1), confusion.sum(axis=0), confusion.diagonal()))


def class_iou(counts):
    target, predicted, correct = counts
    result = np.full(len(target), np.nan)
    present = target > 0
    result[present] = correct[present] / (target + predicted - correct)[present]
    return result


def mean_iou(counts):
    scores = class_iou(counts)
    return float(np.nanmean(scores)) if np.isfinite(scores).any() else None
