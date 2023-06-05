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

# Test ---------------------------------------------------------------------------
print(model.score(X_test, y_test))

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




