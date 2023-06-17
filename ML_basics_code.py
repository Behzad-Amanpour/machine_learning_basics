"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

import numpy as np

# Training-Test Splitting --------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=21)  # Splitting is random but with a fix value for random_state you make it reproducible 

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

# Cross-validation (method2) -----------------------------------------------------
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

# ROC Curve & AUC ----------------------- Behzad Amanpour ------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

model = SVC(kernel='linear', probability=True) # You should use "probability=True" for SVC

y_proba = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]

AUC = roc_auc_score(y_test, y_proba)
print(AUC)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
lw = 2
plt.plot(
    fpr, # shape: 4*1
    tpr, # shape: 4*1
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % AUC, 
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# Standardization & Normalization ------------------------------------------------
from scipy.stats import zscore  # subtracts the mean value, and divids by the standard deviation
X2 = zscore( X )                # calculates zscore in columns
X2 = zscore(X, axis=1)          # calculates zscore in rows

from sklearn.preprocessing import MinMaxScaler     # Transforms features by scaling each feature to the range (0, 1)
scaler = MinMaxScaler()
scaler.fit(X)
X2 = scaler.transform(X)
X2 = MinMaxScaler().fit(X).transform(X)













































