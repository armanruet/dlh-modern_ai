#!/usr/bin/env python3
"""Write a function to generate predictions
with a trained tree-based classifier using Scikit-learn.
"""


def generate_predictions(clf, X):
    """Arguments:
A NumPy array containing the predicted class labels
for the input samples.
    """
    return clf.predict(X)
