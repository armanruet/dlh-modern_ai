#!/usr/bin/env python3
"""Write a function to create a decision tree
classifier using Scikit-learn.
    """
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """ Create a Scikit-learn DecisionTreeClassifier with specified parameters.

    Args:
        min_samples_leaf : Minimum number of samples required at a leaf node.
        min_samples_split : Minimum number of samples required to split.
        random_state : Seed for the random number generator.

    Returns:
        tree_classifier : Configured decision tree classifier.
    """
    tree_classifier = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return tree_classifier
