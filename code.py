"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

import numpy as np

# Training-Test Splitting --------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=21)

# Classification Model -----------------------------------------------------------
from sklearn.svm import SVC
model = SVC(kernel='linear')

# Training -----------------------------------------------------------------------
model.fit(X_train, y_train)

# Test & Prediction --------------------------------------------------------------
print(model.score(X_test, y_test))  # accuracy on the test data
y_pred = model.predict(X_test)      # prediction for the test data

# Cross-validation (method1) -----------------------------------------------------
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=4, scoring='recall')   # cv: The number of parts into which the data is divided
                                                                # scoring = recall / sensitivity / precision / f1 / ...
print(scores)
print(scores.mean())

# Cross-validation (method2) --------------- Behzad Amanpour ---------------------
from sklearn.model_selection import KFold
kf = KFold(n_splits=4, shuffle=False)   # n_splits: The number of parts into which the data is divided
                                        # shuffle: to randomize the order of the data
scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)
print(np.mean(scores))

# Evaluation Metrics ------------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
def specificity_score(y_true, y_pred):  # Scikit-learn has not defined a function for specificity
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]

#  Each function takes two 1-dimensional numpy arrays: the true values of the target & the predicted values of the target.
print("accuracy:", accuracy_score(y_test,y_pred))
print("precision:", precision_score(y_test,y_pred))
print("recall (sensitivity):", recall_score(y_test,y_pred))
print("specificity:", specificity_score(y_test,y_pred))

print("balanced accuracy:", balanced_accuracy_score(y_test,y_pred))
















































