"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

from sklearn.model_selection import KFold
import numpy as np
from sklearn.svm import SVC   # A classification model

kf = KFold( n_splits=len(y), shuffle=False) # shuffle: to randomize the order of the data
y_pred = []; y_true = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred.append( model.predict(X_test) )
    y_true.append( y_test ) # y_true = y  if shuffle=False
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Evaluation Metrics ---------------------------- Behzad Amanpour ----------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support
def specificity_score(y_true, y_pred):   # Scikit-learn has not defined functions for specificity
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]

#  Each function takes two 1-dimensional numpy arrays: the true values of the target & the predicted values of the target.
print("accuracy:", accuracy_score(y_true,y_pred))
print("precision:", precision_score(y_true,y_pred))
print("recall (sensitivity):", recall_score(y_true,y_pred))
print("specificity:", specificity_score(y_true,y_pred))
