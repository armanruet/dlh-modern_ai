#!/usr/bin/env python3
""" Module containing a function to instantiate a Ridge regression model. """
from sklearn import linear_model


def ridge_regression(random_state):
    """ Creates and returns an untrained Ridge Regression model.
    """
    return linear_model.Ridge(random_state=random_state)
