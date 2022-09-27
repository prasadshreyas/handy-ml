#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbors (KNN)

# Objective funtion:
# 
# Find  $f(x) = y_i$ such that,
# 
# $$ i = argmin_{j} ||x - x_j|| $$ 

# **Hyperparameters**
# 
# - $k$: number of nearest neighbors
# - $w$: weight for the neighbors
# - Algorithm for finding the nearest neighbors
# 
# **Time complexity**
# - Fitting: 0
# - Predicting: $O( len(features) \times len(samples) )$

# Extras:
# - KDTree for finding nearest neighbors: 
#   - The number of samples is fixed for the whole training process and the number of features is fixed for each sample.
#   - $O(log(len(samples)) )$
# 

# Example: Iris dataset
# ---

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


X = datasets.load_iris().data
y = datasets.load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))


# References:
# - https://amueller.github.io/COMS4995-s20/slides/
