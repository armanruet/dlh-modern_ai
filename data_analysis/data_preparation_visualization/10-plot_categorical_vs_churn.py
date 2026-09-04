#!/usr/bin/env python3
"""Write a function that visualizes churn rates per category:
"""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """df: pandas DataFrame with Churn column
col: Categorical column name
Returns: None

    Args:
        df (_type_): _description_
        col (_type_): _description_
    """

    plt.figure(figsize=(12, 8))

    # Compute proportion of 'Yes' per category
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean())

    # Create the bar plot
    plt.bar(churn_rate.index, churn_rate.values)

    # Titles and labels
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)

    plt.show()
    return None
