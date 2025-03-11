#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import kFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import kFold,StratifiedkFold
dataframe=pd.read_csv("diabetes.csv")
dataframe


# In[3]:


dataframe=pd.read_csv("diabetes.csv")
dataframe


# In[5]:


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
x = dataframe.iloc[:, 0:8]  
y = dataframe.iloc[:, 8]    
kFold = StratifiedKFold(n_splits=10, random_state=2023, shuffle=True)
model = RandomForestClassifier(n_estimators=200, random_state=20, max_depth=None)
results = cross_val_score(model, x, y, cv=kFold)
print(results)
print("Mean accuracy:", results.mean())


# #### hyper parameter tuning using gridsearchcv

# In[6]:


#use grid search cv to find best parameters
from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier(random_state=42,n_jobs=-1)
params={'max_depth':[2,3,5,None],
        'min_samples_leaf':[5,10,20],
        'n_estimators':[50,100,200,500],
        'max_features':['sqrt','log2',None],
        'criterion':['gini','entropy']
       }
#instantiate the grid search model
grid_search=GridSearchCV(estimator=rf,param_grid=params,cv=5,n_jobs=-1,verbose=10,scoring='accuracy')
grid_search.fit(x,y)


# In[12]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[13]:


grid_search.best_estimator_


# #### feature slection using random forest 
# 

# In[17]:


model_best = RandomForestClassifier(criterion='entropy', max_depth=5, max_features=None,
                       min_samples_leaf=5, n_jobs=-1, random_state=42)
model_best.fit(x,y)
model_best.feature_importances_


# In[15]:


X = dataframe.iloc[:,0:8]
X.columns


# In[18]:


df=pd.DataFrame(model_best.feature_importances_,columns=['importance score'],index=x.columns)
df.sort_values(by='importance score',inplace=True,ascending=False,)


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.bar(df.index,df['importance score'])


# In[ ]:





# In[ ]:




