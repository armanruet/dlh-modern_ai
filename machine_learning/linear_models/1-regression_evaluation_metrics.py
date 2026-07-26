#!/usr/bin/env python3
"""compute common evaluation metrics """
from sklearn import metrics
import numpy as np


def evaluation_metrics_for_regression(y_true, y_pred):
    """ Computes common regression evaluation metrics: MSE, RMSE, MAE, and R².

    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    return mse, rmse, mae, r2
