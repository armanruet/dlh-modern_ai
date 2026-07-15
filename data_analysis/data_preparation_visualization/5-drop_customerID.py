#!/usr/bin/env python3
"""Task 5"""


def remove_duplicates(df):
    """def the func"""
    df = df.drop(["customerID"])
    return df


