#!/usr/bin/env python
import numpy as np
import pandas as pd

# Models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Pre processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Helper files
from common import test_env
from common.describe_data import (print_categorical, print_overview,
                                  print_nan_counts)
from common.classification_metrics import print_metrics

# Various definitions
OVERVIEW_FILE = 'results/students_{}overview.txt'
CAT_FILE = 'results/students_{}categorical_data.txt'
MODELS = {
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(gamma="auto"),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Logistic Regression": LogisticRegression(solver="lbfgs"),
}


def read_data(file):
    """Return pandas dataFrame read from XLSX file"""
    try:
        return pd.read_excel(file)
    except FileNotFoundError:
        exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    """Process our data, encoding categorical columns and removing NaNs
    """
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

    """5. Save dataset description and categorical features to file
    in results directory."""
    print_overview(df, file=OVERVIEW_FILE.format(""))
    print_categorical(df, file=CAT_FILE.format(""))

    """6. Filter out students not continuing studies
            (In university aster 4 semesters is No) and save filtered
            description and categorical features to file."""
    idx = y == 0
    print_overview(df[idx],
                   file=OVERVIEW_FILE.format("not_continuing_"))
    print_categorical(df[idx],
                      file=CAT_FILE.format("not_continuing_"))

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    """8. Preprocess data by encoding categorical features and
            by replacing missing numeric exam points with 0"""
    # STUDENT SHALL ENCODE CATEGORICAL FEATURES
    # Fill all NaN in categorical cols as "Missing"
    df.fillna({x: 'Missing' for x in categorical_columns}, inplace=True)
    # Encode
    dummies = pd.get_dummies(df[categorical_columns])
    df.drop(columns=categorical_columns, inplace=True)
    df[dummies.columns] = dummies

    # Handle missing data. At this point only exam points should be missing
    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        print_nan_counts(df)

    # STUDENT SHALL HANDLE MISSING VALUES
    df.fillna(value=0, inplace=True)
    if verbose:
        print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# STUDENT SHALL CREATE FUNCTIONS FOR LOGISTIC REGRESSION CLASSIFIER, KNN
# CLASSIFIER, SVM CLASSIFIER, NAIVE BAYES CLASSIFIER, DECISION TREE
# CLASSIFIER AND RANDOM FOREST CLASSIFIER
def run_model(x, y, model, model_name, scale=True, test_size=0.25,
              verbose=False):
    """Runs a full experiment with a given model in our data
        and log the results with print_metrics
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y,
                                                        test_size=test_size)
    if scale:  # scale data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Fit model and print accuracy
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print_metrics(y_test, y_pred, model_name, verbose=verbose)
    print("\n\n")


if __name__ == '__main__':
    # Set random seed for reproducing results
    np.random.seed(42)

    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')

    # STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    # FILE NAME AS ARGUMENT
    students_x, students_y = preprocess_data(students)

    # STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS
    for model_name, model in MODELS.items():
        run_model(students_x, students_y, model, model_name,
                  verbose=True)

    print('Done')
