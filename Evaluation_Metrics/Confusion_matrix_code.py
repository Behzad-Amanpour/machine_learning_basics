"""
Inputs:
    X_train: n*m  numerical matrix which "n" is the number of samples, and "m" is the number of features
    X_test:  k*m  numerical matrix
    y_train: n*1  array which has the labels of the rows in X_train
    y_train: k*1  array which has the labels of the rows in X_test
"""

# Model Training & Prediction ----------------------------------------------------
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion Matrix ---------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

labels = ['negative', 'positive']
cm = confusion_matrix( y_test, y_pred )
ConfusionMatrixDisplay( cm, display_labels = labels ).plot()
