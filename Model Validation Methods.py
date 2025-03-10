#!/usr/bin/env python
# coding: utf-8

# #### !.Evaluate using a train and a test set

# In[3]:


#Evaluate using atrain and a test set
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[4]:


data = read_csv('diabetes.csv')
data


# In[5]:


data.info()


# #### Model validation using train_test_split()

# In[12]:


# Split the data into train test sets and find the test accuracy

array = data.values
X = array[:,0:8]
Y = array[:,8]

X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 3)
model1 = DecisionTreeClassifier()
model1.fit(X_train, Y_train)
y_predict = model1.predict(X_test)
print(classification_report(y_test, y_predict))


# #### 2.Evaluate using K-fold Cross Validation

# In[14]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[16]:


X = array[:,0:8]
Y = array[:,8]
kfold = KFold(n_splits=7)
model2 = DecisionTreeClassifier()
results2 = cross_val_score(model2, X, Y, cv=kfold)
print(results2)


# In[18]:


print(results2.mean())


# In[19]:


results2.std()


# #### 3.Evaluate using Leave one out Cross validation

# In[20]:


from sklearn.model_selection import LeaveOneOut

array = data.values
X = array[:,0:8]
Y = array[:,8]
loocv = LeaveOneOut()
model3 = DecisionTreeClassifier()
results3 = cross_val_score(model3, X, Y, cv=loocv)
results3


# In[21]:


results3.mean()


# In[22]:


results3.std()


# In[ ]:




