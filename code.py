"""
Inputs:
    X: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    y: n*1  array which has the labels of rows in X
"""


# Training-Test Splitting ------------------- Behzad Amanpour --------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=21)

# Classification Model
from sklearn.svm import SVC
model = SVC(kernel='linear')

# Training
model.fit(X_train, y_train)

# Test
print(model.score(X_test, y_test))
