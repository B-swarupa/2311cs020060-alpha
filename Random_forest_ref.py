#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import kFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import kFold,StratifiedkFold
dataframe=pd.read_csv("diabetes.csv")
dataframe


# In[2]:


dataframe=pd.read_csv("diabetes.csv")
dataframe


# In[10]:


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
x = dataframe.iloc[:, 0:8]  
y = dataframe.iloc[:, 8]    
kFold = StratifiedKFold(n_splits=10, random_state=2023, shuffle=True)
model = RandomForestClassifier(n_estimators=200, random_state=20, max_depth=None)
results = cross_val_score(model, x, y, cv=kFold)
print(results)
print("Mean accuracy:", results.mean())


# In[ ]:





# In[ ]:




