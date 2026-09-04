#!/usr/bin/env python3
"""Write a function that engineers new features from the dataset
    """


import pandas as pd


def create_features(df):
    """df: pandas DataFrame


Returns the modified DataFrame

    Args:
        df (_type_): _description_
    """
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV',
                'StreamingMovies', 'MultipleLines']

    df = pd.DataFrame(df)  # to access function helpers, DF
    # Count 'Yes' across service columns
    df['NumServices'] = (df[services] == 'Yes').sum(axis=1)
    # Add 1 if InternetService is DSL or Fiber optic
    df['NumServices'] += df['InternetService'].isin(
        ['DSL', 'Fiber optic']).astype(int)
    # pd.cut is the standard way to create bins
    bins = [0, 12, 24, 48, 60,  df['tenure'].max() + 1]
    labels = ['0-12', '13-24', '25-48', '49-60', '60+']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels,
                               right=True, include_lowest=False)
    # drop source columns used to create the new features
    df = df.drop(columns=services + ['InternetService', 'tenure'])
    return df
