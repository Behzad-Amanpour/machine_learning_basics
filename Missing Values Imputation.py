# KNN Imputer ================== Behzad Amanpour ==============================
from sklearn.impute import KNNImputer
Imputer = KNNImputer() # n_neighbors = 3
X2 = Imputer.fit_transform( X ) 

# IterativeImputer ============= Behzad Amanpour ==============================
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
Imputer = IterativeImputer() # random_state=0
X3 = Imputer.fit_transform( X )
X3 = np.round( Imputer.fit_transform( X ) )  # might be needed
