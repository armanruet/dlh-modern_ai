#!/usr/bin/env python3
"""
Task 15: Encode features
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Encodes features for modeling using Scikit-learn.

    Args:
        df: pandas DataFrame

    Returns:
        df_enc: the encoded DataFrame
        churn_le: fitted LabelEncoder for Churn
        binary_oe: fitted OrdinalEncoder for binary columns
        tenure_oe: fitted OrdinalEncoder for TenureGroup
    """
    df_enc = df.copy()

    # Churn: LabelEncoder (No -> 0, Yes -> 1)
    churn_le = preprocessing.LabelEncoder()
    df_enc['Churn'] = churn_le.fit_transform(df_enc['Churn'])

    # Binary columns: OrdinalEncoder (No -> 0, Yes -> 1)
    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling', 'SeniorCitizen']
    binary_oe = preprocessing.OrdinalEncoder()
    df_enc[binary_cols] = binary_oe.fit_transform(
        df_enc[binary_cols]).astype(int)

    # TenureGroup: alphabetical order OrdinalEncoder
    tenure_oe = preprocessing.OrdinalEncoder()
    df_enc[['TenureGroup']] = tenure_oe.fit_transform(
        df_enc[['TenureGroup']]).astype(int)

    # Contract, PaymentMethod: one-hot encoding, drop first category
    df_enc = pd.get_dummies(
        df_enc, columns=['Contract', 'PaymentMethod'],
        drop_first=True, dtype=int)

    return df_enc, churn_le, binary_oe, tenure_oe
