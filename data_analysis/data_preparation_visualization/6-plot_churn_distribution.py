#!/usr/bin/env python3
"""
Task 6: Plot Target Distribution
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Visualizes the churn class distribution as a bar plot.

    Args:
        df: pandas DataFrame with a Churn column

    Returns:
        None
    """
    counts = df['Churn'].value_counts().reindex(['No', 'Yes'])

    plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'])
    plt.title('Churn Distribution')
    plt.ylabel('Count')
    plt.show()

    return None
