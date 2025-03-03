#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[22]:


iris=pd.read_csv("iris.csv")
iris


# In[23]:


import seaborn as sns
counts=iris["variety"].value_counts()
sns.barplot(data=counts)


# In[24]:


iris.info()


# In[25]:


iris[iris.duplicated(keep=False)]


# #### observations
# - there are 150 rows and 5 columns
# - there are no null values
# - there is one du

# In[26]:


iris =iris.drop_duplicates(keep='first')
iris[iris.duplicated]


# In[27]:


#reset the indexx
iris=iris.reset_index(drop=True)
iris


# In[28]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()
                                           


# In[29]:


iris.info()


# #### observation
# - the target column(variety)is still object type.it needs to be converted to numeric(int)

# In[30]:


#converte the target column data type to integer
iris['variety']=pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[31]:


#divide the dataset into x-columns and y-coulumns
x=iris.iloc[:,0:4]
y=iris['variety']
y


# In[32]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1)
x_train


# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1)
x_train.head(20)


# In[34]:


from sklearn.model_selection import train_test_split
model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[38]:


#plot the decision tree
plt.figure(dpi=1200)
tree.plot_tree(model);


# In[39]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[41]:


# Predicting on test data
preds = model.predict(x_test)# predicting on test data set
preds


# In[42]:


print(classification_report(y_test,preds))


# In[ ]:




