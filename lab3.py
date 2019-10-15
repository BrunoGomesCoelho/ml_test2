#!/usr/bin/env python
import numpy as np
import pandas as pd

from common import describe_data, test_env


def read_data(file):
    """Return pandas dataFrame read from csv file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # STUDENT SHALL ENCODE CATEGORICAL FEATURES

    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # STUDENT SHALL HANDLE MISSING VALUES

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# STUDENT SHALL CREATE FUNCTIONS FOR LOGISTIC REGRESSION CLASSIFIER, KNN
# CLASSIFIER, SVM CLASSIFIER, NAIVE BAYES CLASSIFIER, DECISION TREE
# CLASSIFIER AND RANDOM FOREST CLASSIFIER


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')

    # STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    # FILE NAME AS ARGUMENT

    students_X, students_y = preprocess_data(students)

    # STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS

    print('Done')
