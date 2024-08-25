import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def data_splitting(data_frame, feature_selection, target_column):
    cleaned_data_frame = data_frame.dropna(subset=target_column)
    X = cleaned_data_frame[feature_selection]
    y = cleaned_data_frame[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def logistic_regression(X_train, X_test, y_train, y_test, feature):
    log_reg = LogisticRegression()
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])

    # Calculate F1 score, the harmonic mean of precision and recall
    # Could also use balanced_accuracy_score
    f1 = metrics.f1_score(y_test, y_pred, average = "weighted")

    return f1