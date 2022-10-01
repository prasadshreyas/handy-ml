#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning and Model Evaluation

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


# ## Precision, Recall, F1 Score
# 
# Precision, recall, and F1 score are metrics for classification problems. They are defined as follows:
# 
# - Precision: The number of true positives divided by the number of true positives plus the number of false positives.
#   $$\text{precision} = \frac{TP}{TP + FP}$$
# - Recall: The number of true positives divided by the number of true positives plus the number of false negatives.
#   $$\text{recall} = \frac{TP}{TP + FN}$$
# - F1 score: The harmonic mean of precision and recall.
#   $$ \text{F1 score} = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$
# 
# 
# 
# 
# ## ROC Curve and AUC
# 
# The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The area under the curve (AUC) is a measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, higher the AUC, better the model is at distinguishing between patients with disease and no disease.
# 
# - x-axis: False Positive Rate (FPR)
# - y-axis: True Positive Rate (TPR) = Recall
# 
# 
# 
# 
# 
# 
# 

# ## Residual Plots
# 
# Used to check if the model is linearly related to the target variable.
# 
# - Residuals are the difference between the observed value of the target variable (y) and the predicted value (ŷ).
# 

# 
