from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """Classification accuracy"""
    assert y_hat.size == y.size, "Predictions and labels must be same length"
    if y.size == 0:
        return 0.0
     # Reset index to align both series and count correct matches
    return (y_hat.reset_index(drop=True) == y.reset_index(drop=True)).sum() / y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """Precision = TP / (TP + FP)"""
    assert y_hat.size == y.size, "Predictions and labels must be same length"
    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """Recall = TP / (TP + FN)"""
    assert y_hat.size == y.size, "Predictions and labels must be same length"
    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """Mean Absolute Error for regression"""
    assert y_hat.size == y.size, "Size of y_hat and y must be equal."
    assert y.size, "Ground Truth array is 0"
    assert y_hat.size, "Predicition array is 0"

    y_c = y.copy()
    y_hat_c = y_hat.copy()

    y_hat_c = np.array(y_hat_c)
    y_c = np.array(y_c)
    numerator = np.sum((np.abs(y_hat_c-y_c)))

    denominator = y_c.size

    return numerator/denominator

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """Root Mean Squared Error for regression UNDERSTAND THIS"""
    assert y_hat.size == y.size, "Predictions and labels must be same length"
    assert y.size!=0
    assert y_hat.size!=0
    y_c = y.copy()
    y_hat_c = y_hat.copy()

    y_hat_c = np.array(y_hat_c)
    y_c = np.array(y_c)
    numerator = np.sum((y_hat_c-y_c)**2)
    denominator = y_c.size

    return np.sqrt(numerator/denominator) 

