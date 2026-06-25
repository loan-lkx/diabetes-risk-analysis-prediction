import pandas as pd
import numpy as np

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def load_raw_data(path):
    return pd.read_csv(path)


def load_cleaned_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()
    for col in ZERO_INVALID_COLS:
        df[col] = df[col].replace(0, np.nan)

    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean()).round().astype(int)
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean()).round().astype(int)
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median()).astype(int)
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median()).astype(int)
    df['BMI'] = df['BMI'].fillna(df['BMI'].median()).round(1)

    return df


def detect_outliers_iqr(df, features=None):
    if features is None:
        features = FEATURES
    results = {}
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        results[col] = {
            'lower': lower,
            'upper': upper,
            'count': len(outliers),
            'percent': round(len(outliers) / len(df) * 100, 2)
        }
    return results
