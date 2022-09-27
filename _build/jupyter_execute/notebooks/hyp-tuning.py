#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning

# We use **cross-validation** to tune the hyperparameters of a model.
# Some of the common techniques are:
# - Grid Search
# - Shuffle Split
# - Stratified Shuffle Split
# - Group K Fold

# ## Grid Search

# GridSearchCV does an exhaustive search over specified parameter values for an estimator. The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# In[4]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X = load_iris().data
y = load_iris().target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_neighbors': np.arange(1, 10, 2)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5 , return_train_score=True)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
print(grid.score(X_test, y_test))


# In[5]:


grid


# In[7]:


import pandas as pd
pd.DataFrame(grid.cv_results_)


# ## KFold, ShuffleSplit, StratifiedShuffleSplit, GroupKFold

# In[9]:


from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5)
skfold = StratifiedKFold(n_splits=5)
ss = ShuffleSplit(n_splits=5, test_size=0.2)
rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

print("KFold:\n", cross_val_score(KNeighborsClassifier(), X, y, cv=kfold))
print("StratifiedKFold:\n", cross_val_score(KNeighborsClassifier(), X, y, cv=skfold))
print("ShuffleSplit:\n", cross_val_score(KNeighborsClassifier(), X, y, cv=ss))
print("RepeatedStratifiedKFold:\n", cross_val_score(KNeighborsClassifier(), X, y, cv=rskfold))


# In[ ]:




