#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


iris=pd.read_csv("iris.csv")
iris


# In[5]:


import seaborn as sns
counts=iris["variety"].value_counts()
sns.barplot(data=counts)


# In[6]:


iris.info()


# In[7]:


iris[iris.duplicated(keep=False)]


# In[ ]:


#### observations
- there are 150 rows and 5 columns
- there are no null values
- there is one du


# In[13]:


iris =iris.drop_duplicates(keep='first')
iris[iris.duplicated]


# In[9]:


#reset the indexx
iris=iris.reset_index(drop=True)
iris


# In[11]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
iris.iloc[:,-1]=labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()
                                           


# In[14]:


iris.info()


# #### observation
# - the target column(variety)is still object type.it needs to be converted to numeric(int)

# In[15]:


#converte the target column data type to integer
iris['variety']=pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[20]:


#divide the dataset into x-columns and y-coulumns
x=iris.iloc[:,0:4]
y=iris['variety']
y


# In[23]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1)
x_train


# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 1)
x_train.head(20)


# In[ ]:





# In[ ]:


#plot the decision tree
plt.figure(dpi=1200)
tree.plot_tree(model);

