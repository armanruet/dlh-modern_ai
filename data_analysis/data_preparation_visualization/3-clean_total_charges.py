#!/usr/bin/env python3
"""Task 3"""


def clean_total_charges(df, method='drop'):
    """Def the func"""
    df = df.copy()

    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        median_val = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    elif method == 'impute':
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['MonthlyCharges'] * df['tenure'])

    return df
