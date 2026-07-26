#!/usr/bin/env python3
""" Module containing a function to instantiate a Lasso regression model. """

from sklearn import linear_model


def lasso_regression(random_state):
    """ Creates and returns an untrained Lasso Regression model.

    """
    return linear_model.Lasso(random_state=random_state)
