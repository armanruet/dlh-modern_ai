#!/usr/bin/env python3
"""Write a function to create a random forest
classifier using Scikit-learn.
    """
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """Arguments:

    model: A Scikit-learn RandomForestClassifier instance.
    """
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    return model
