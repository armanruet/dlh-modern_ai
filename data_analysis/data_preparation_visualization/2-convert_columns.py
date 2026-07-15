#!/usr/bin/env python3
"""Task 2"""
import pandas as pd


def convert_columns(df):
    """Def the func"""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df
