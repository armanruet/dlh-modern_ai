#!/usr/bin/env python3
"""Linear Reg"""
from sklearn import linear_model


def Linear_Regression():
    """ Creates and returns an untrained LinearRegression.
    Returns:
        LinearRegression: An instance of sklearn.linear_model.
    """
    return linear_model.LinearRegression()
