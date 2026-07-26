#!/usr/bin/env python3
""" Module containing a function to generate SHAP explainer / SHAP values. """
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """ Creates a SHAP explainer using X_train as the background dataset and
    computes SHAP values for X_test.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
