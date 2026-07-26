#!/usr/bin/env python3
""" Module containing a function to create SVM classifier using Scikit-learn
with the specified kernel.
"""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Creates and returns an untrained SVM classifier with the specified kernel.
   """
    kernel = {'linear', 'poly', 'rbf'}
    if name not in kernel:
        raise ValueError(f"Invalid kernel. Expected one of {kernel}.")

    return svm.SVC(kernel=name, random_state=random_state)
