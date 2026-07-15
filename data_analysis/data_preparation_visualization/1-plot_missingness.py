#!/usr/bin/env python3
"""task 1"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """def the function"""
    plt.figure(figsize=(12, 8))

    x_vals = []
    y_vals = []
    for i, col in enumerate(df.columns):
        missing_idx = df.index[df[col].isnull()]
        x_vals.extend(missing_idx)
        y_vals.extend([i] * len(missing_idx))

    plt.scatter(x_vals, y_vals, marker='|')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title("Missingness Plot")

    plt.tight_layout()
    plt.show()
