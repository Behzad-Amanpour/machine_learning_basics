"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

# Classification model ================================================================
from sklearn.linear_model import LogisticRegression
model = LogisticRegression( solver = 'liblinear' )

from sklearn.svm import SVC
model = SVC(kernel="linear")

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, random_state=10)

# RFE ================================ Behzad Amanpour ===============================
from sklearn.feature_selection import RFE
import numpy as np

fs_model = RFE(
    estimator = model,
    step = 1,  # number of features to remove at each iteration
    n_features_to_select = 5, # the number of features to consider
    )

fs_model.fit(X, y)
selected = fs_model.ranking_
ix = np.where( selected ==1 )[0]  # ix=[1,5]
scores = cross_val_score(model, X[:,ix], y, cv=5, scoring = metric) 
print(scores.mean())

# RFECV =============================== Behzad Amanpour ===============================
from sklearn.feature_selection import RFECV
import numpy as np
fs_model = RFECV(
    estimator = model,
    step = 1,  # number of features to remove at each iteration
    scoring = metric,
    min_features_to_select = 2, # Minimum number of features to consider
    )
fs_model.fit(X, y)
selected = fs_model.ranking_
ix = np.where(selected==1)[0] 
scores = cross_val_score(model, X[:,ix], y, cv=5, scoring = metric) 
print(scores.mean())
