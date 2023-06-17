"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

import numpy as np

# Evaluation Metrics ------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
def specificity_score(y_true, y_pred):  # Scikit-learn has not defined a function for specificity
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]

print("accuracy:", accuracy_score(y_test,y_pred))
print("precision:", precision_score(y_test,y_pred))
print("recall (sensitivity):", recall_score(y_test,y_pred))
print("specificity:", specificity_score(y_test,y_pred))
print("balanced accuracy:", balanced_accuracy_score(y_test,y_pred))
