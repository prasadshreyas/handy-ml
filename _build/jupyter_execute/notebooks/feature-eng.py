#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering
# 

# ## Scaling:
# 
# Scale for all models which use distance based metrics (KNN, SVM, etc.)
# 
# **Types of Scaling**:
# - Standardization: mean = 0, std = 1
# - Min-Max Scaling: min = 0, max = 1

# ## Encoding
# 
# **Types of Encoding**:
# - One-Hot Encoding
#   - $R \rightarrow \{1, 0\}$
#   - Creates a new column for each category
# - Ordinal Encoding
#   - $ categories \rightarrow \{1, 2, 3, \dots\}$
#   - Does not create new columns (only one column)
# - Binary Encoding
#   - $ categories \rightarrow \{00, 01, 10, 11, \dots\}$
# - Binning
#   - $ set(n) \rightarrow  len(\{0, 1, 2, \dots\}) < len(set(n))$

# ## Pre-processing
# 
# ### Missing Values
#   - Drop
#   - Impute
#     - Mean
#     - Median
#     - Mode
#     - KNN
#     - MICE

# To-do:
# 
# - [ ] Mean vs Median
#   - Mean is sensitive to outliers, so use median for skewed data

# 
