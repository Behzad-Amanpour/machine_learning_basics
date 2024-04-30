# Evaluation Metrics ------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("accuracy: ", accuracy_score( y_test, y_pred) )
print("precision: ", precision_score( y_test, y_pred) )
print("recall (sensitivity): ", recall_score( y_test, y_pred) )
print("recall (specificity): ", recall_score( y_test, y_pred, pos_label=0) ) # average != 'binary'
print("F1_score_1: ", f1_score( y_test, y_pred) )
print("F1_score_0: ", f1_score( y_test, y_pred, pos_label=0) ) # average != 'binary'

from sklearn.metrics import balanced_accuracy_score
print("balanced accuracy:", balanced_accuracy_score(y_test,y_pred))


# ----------------------------------- Behzad Amanpour --------------------
from sklearn.metrics import classification_report
print( classification_report(y_test, y_pred) )

