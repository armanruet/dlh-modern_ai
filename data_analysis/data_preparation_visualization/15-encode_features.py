#!/usr/bin/env python3
"""function encoding features"""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    df: pandas DataFrame
    """
    df_copy = df.copy()
    churn_le = preprocessing.LabelEncoder()
    df_enc['Churn'] = churn_le.fit_transform(df_enc['Churn'])

    bin_cols = ['Partner', 'Dependents', 'PaperlessBilling', 'SeniorCitizen']
    binary_oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']])
    for c in bin_cols:
        df_enc[c] = binary_oe.fit_transform(df_enc[[c]])
        df_enc[c] = df_enc[c].astype(int)

    df_enc = pd.get_dummies(
        df_enc,
        columns=['Contract', 'PaymentMethod'], drop_first=True, dtype=int
    )

    tenure_oe = preprocessing.OrdinalEncoder()

    df_enc['TenureGroup'] = tenure_oe.fit_transform(df_enc[['TenureGroup']])
    df_enc['TenureGroup'] = df_enc['TenureGroup'].astype(int)

    return df_enc, churn_le, binary_oe, tenure_oe
