#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold


# In[5]:


dataframe=pd.read_csv("diabetes.csv")
dataframe


# In[6]:


array=dataframe.values
x=array[:,0:8]
y=array[:,8]


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y)


# In[8]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[9]:


clf = SVC()
clf.fit(x_train,y_train)


# In[10]:


y_predict=clf.predict(x_test)


# In[11]:


print(classification_report(y_test,y_predict))


# In[ ]:


clf=SVC()
param_grid=[{'kernel':['linear','rbf'],'gamma':[0.1,0.5,1],'C':[0.1,1,10]}]
kfold=StratifiedKFold(n_splits=5)
gsv=RandomizedSearchCV(clf,param_grid,cv=kfold,scoring='recall')
gsv.fit(x_train,y_train)            


# In[ ]:


gsv.best_params_,gsv.best_score_


# In[ ]:


clf_model=SVC(kernel='linear',C=1)
clf_model.fit(x_train,y_train)
y_pred=clf_model.predict(x_test)
acc=accuracy_score(y_test,y_pred)*100
print("accuracy=",acc)
confusion_matrix(y_test,y_pred)


# In[ ]:


y_Pred


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classfication_report(y_test,y_pred))


# In[ ]:




