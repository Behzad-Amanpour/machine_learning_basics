"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""

# Creating a classifier model ========== Behzad Amanpour ===========================
from sklearn.linear_model import LogisticRegression as LR
model = LR()

from sklearn.neighbors import KNeighborsClassifier as knn
model = knn() # n_neighbors = 3

from sklearn import svm
model = svm.SVC(kernel="linear", probability=True)

from sklearn.ensemble import RandomForestClassifier as RF
model = RF() # RF(n_estimators=200, random_state=0)


# ROC Curve & AUC ====================== Behzad Amanpour ==========================
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

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
