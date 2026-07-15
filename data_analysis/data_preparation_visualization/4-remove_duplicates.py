#!/usr/bin/env python3
"""Task 4"""


def remove_duplicates(df):
    """def the func"""
    df = df.copy()
    df = df.drop_duplicates()
    return df
