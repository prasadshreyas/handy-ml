#!/usr/bin/env python
# coding: utf-8

# # Linear Models
# 
# Table of Contents
# 1. Linear Regression
#    1. OLS
#    2. Lasso Regression (L1)
#    3. Ridge Regression (L2)
# 2. Logistic Regression
# 
# 
# [ ] - How does one hot encoding introduce collinearity? (last one )

# ## Linear Regression

# ### OLS (Ordinary Least Squares)
# 
# OLS is a linear regression model that minimizes the sum of squared errors. It is the most common method of linear regression. It is also known as the least squares method.
# 
# $$ min_{w \in \mathbb{R}^p, b \in \mathbb{R}} \sum_{i=1}^n (w^T x_i + b - y_i)^2 $$
# 
# where $\hat{y}_i$ is the predicted value.

# In[27]:



import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


# one dimensional data normally distributed around 0 with a standard deviation of 1
X = np.random.normal(0, 1, 100)
# y = 2x + 3
y = 2 * X + 3
# add noise to the data
y += np.random.normal(0, 1, 100)

model = LinearRegression( )
model.fit(X.reshape(-1, 1), y)

# MSE
print('MSE:', np.mean((model.predict(X.reshape(-1, 1)) - y) ** 2))


# Plot the data and the regression line
plt.figure( figsize=(3,3), dpi=150) 
sns.set_style('darkgrid')
plt.scatter(X, y, color='blue', marker='o', s=10)
plt.title('OLS', fontsize=10)
plt.xlabel('X')
plt.ylabel('y')
plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', linewidth=0.5)
plt.show()


# OLS is not robust to outliers. It is also sensitive to the scale of the data. See the example below.

# In[28]:


# Add some outliers
X = np.random.normal(0, 1, 25)
y = 2 * X + 3
y += np.random.normal(0, 0.5, 25)
X = np.append(X, [1.1, 1.2, 1.3])
y = np.append(y, [30, 31, 32])

# Co-efficients
model = LinearRegression( )
model.fit(X.reshape(-1, 1), y)

# MSE
print('MSE:', np.mean((model.predict(X.reshape(-1, 1)) - y) ** 2))


# Plot the data and the regression line
plt.figure( figsize=(3,3), dpi=150) 
sns.set_style('darkgrid')
plt.scatter(X, y, color='blue', marker='o', s=10)
plt.title('OLS', fontsize=10)
plt.xlabel('X')
plt.ylabel('y')
plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', linewidth=0.5)
plt.show()


# 

# ### Lasso Regression (L1)

# ### Ridge Regression (L2)

# ## Logistic Regression

# In[ ]:





# In[ ]:




